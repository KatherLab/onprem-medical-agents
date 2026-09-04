import asyncio
import json
import logging
from pathlib import Path

from openai import AsyncOpenAI

from .evaluation_input import EvaluationInput
from .metrics import DiagnosisMetrics
from .prompt_builders import PromptBuilder
from .utils import response_parse

logger = logging.getLogger(__name__)

class AsyncLLMClient:
    def __init__(self, client: AsyncOpenAI, model: str, temperature: float = 0.01, max_tokens: int = 1024):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def query(self, prompt: str) -> str:  # Must be asynchronous.
        response = await self.client.chat.completions.create(  # Await the async client call.
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()


class DiagnosisEvaluator:
    def __init__(self, client: AsyncOpenAI, model, match_criterion: str, output_path: str):
        self.match_criterion = match_criterion
        self.prompt_builder = PromptBuilder(self.match_criterion)
        self.llm_client = AsyncLLMClient(client, model)  # Use the asynchronous client.
        self.metrics = DiagnosisMetrics(output_path)
        self.output_path = output_path
        self._lock = asyncio.Lock()

    @staticmethod
    def _get_input_identifier(input_data: EvaluationInput) -> str:
        """Support both legacy hadm_id-based inputs and VivaBench case_id-based inputs."""
        return str(getattr(input_data, "hadm_id", getattr(input_data, "case_id", "unknown_case")))

    def _build_batch_custom_id(self, input_data: EvaluationInput) -> str:
        return f"case-{self._get_input_identifier(input_data)}"

    def _build_batch_request(self, input_data: EvaluationInput) -> tuple[str, dict, dict] | None:
        final_outputs = input_data.assistant_diagnosis()
        ad = final_outputs.get('assistant_diagnosis')
        prob = final_outputs.get('geometric_mean_probability')
        reason = final_outputs.get('assistant_reasoning')
        rprob = final_outputs.get('reason_geometric_mean_probability')
        gt = input_data.ground_truth_diagnosis(self.match_criterion)

        if not gt or not ad:
            return None

        prompt = self.prompt_builder.build(gt, ad, reason)
        custom_id = self._build_batch_custom_id(input_data)
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.llm_client.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.llm_client.temperature,
                "max_tokens": self.llm_client.max_tokens,
            },
        }
        context = {
            "hadm_id": self._get_input_identifier(input_data),
            "ground_truth": gt,
            "assistant_diagnosis": ad,
            "geometric_mean_probability": prob,
            "assistant_reasoning": reason,
            "reason_geometric_mean_probability": rprob,
        }
        return custom_id, request, context

    def write_batch_requests_file(self, inputs: list[EvaluationInput], request_path: str) -> dict[str, dict]:
        contexts: dict[str, dict] = {}
        request_file = Path(request_path)
        request_file.parent.mkdir(parents=True, exist_ok=True)

        with request_file.open("w", encoding="utf-8") as f:
            for input_data in inputs:
                batch_row = self._build_batch_request(input_data)
                if batch_row is None:
                    continue

                custom_id, request, context = batch_row
                contexts[custom_id] = context
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        return contexts

    @staticmethod
    def _extract_batch_content(result_row: dict) -> str | None:
        response = result_row.get("response") or {}
        body = response.get("body") or {}

        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_chunks = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                return "".join(text_chunks).strip() or None

        output = body.get("output") or []
        text_chunks = []
        for item in output:
            for part in item.get("content", []):
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                    text_chunks.append(part.get("text", ""))
        if text_chunks:
            return "".join(text_chunks).strip()

        return None

    def consume_batch_results(self, result_text: str, contexts: dict[str, dict]):
        for line in result_text.splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                result_row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed batch output line: {e}")
                continue

            custom_id = result_row.get("custom_id")
            if not custom_id:
                logger.warning("Skipping batch output row without custom_id.")
                continue

            context = contexts.get(custom_id)
            if context is None:
                logger.warning(f"Skipping batch output row with unknown custom_id: {custom_id}")
                continue

            error = result_row.get("error")
            if error:
                logger.error(f"Batch request failed for {custom_id}: {error}")
                continue

            llm_response = self._extract_batch_content(result_row)
            if not llm_response:
                logger.warning(f"Skipping {custom_id} because batch output had no assistant content.")
                continue

            parsed_result = response_parse(llm_response)
            if parsed_result is None:
                logger.warning(
                    f"Skipping HADM_ID {context['hadm_id']} due to LLM response parsing failure. "
                    f"Response was: '{llm_response[:100]}...'"
                )
                continue

            self.metrics.add_result(
                context["hadm_id"],
                context["ground_truth"],
                context["assistant_diagnosis"],
                context["geometric_mean_probability"],
                context["assistant_reasoning"],
                context["reason_geometric_mean_probability"],
                parsed_result,
            )

    async def evaluate_batch(
        self,
        inputs: list[EvaluationInput],
        request_path: str,
        poll_interval: int = 30,
        metadata: dict | None = None,
    ) -> str | None:
        contexts = self.write_batch_requests_file(inputs, request_path)
        if not contexts:
            logger.warning("No valid diagnosis evaluation inputs were available for batch submission.")
            return None

        with open(request_path, "rb") as f:
            input_file = await self.llm_client.client.files.create(file=f, purpose="batch")

        batch = await self.llm_client.client.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=input_file.id,
            metadata=metadata or {},
        )
        logger.info(f"Submitted batch job {batch.id} with {len(contexts)} requests.")

        terminal_statuses = {"completed", "failed", "expired", "cancelled"}
        while batch.status not in terminal_statuses:
            logger.info(f"Batch {batch.id} status: {batch.status}. Polling again in {poll_interval}s.")
            await asyncio.sleep(poll_interval)
            batch = await self.llm_client.client.batches.retrieve(batch.id)

        logger.info(f"Batch {batch.id} reached terminal status: {batch.status}")
        if batch.status != "completed":
            logger.error(f"Batch {batch.id} did not complete successfully.")
            return batch.id

        if not batch.output_file_id:
            logger.error(f"Batch {batch.id} completed without an output_file_id.")
            return batch.id

        result_text = await self.llm_client.client.files.retrieve_content(batch.output_file_id)
        self.consume_batch_results(result_text, contexts)
        return batch.id

    async def evaluate_single(self, input_data: EvaluationInput):  # Evaluate one input asynchronously.
        final_outputs = input_data.assistant_diagnosis()
        ad = final_outputs.get('assistant_diagnosis')
        prob = final_outputs.get('geometric_mean_probability')
        reason = final_outputs.get('assistant_reasoning')
        rprob = final_outputs.get('reason_geometric_mean_probability')
        gt = input_data.ground_truth_diagnosis(self.match_criterion)
        input_identifier = self._get_input_identifier(input_data)

        if not gt or not ad:
            return 

        prompt = self.prompt_builder.build(gt, ad, reason)
        # Query and parse the response here; aggregate results externally.
        try:
            llm_response = await self.llm_client.query(prompt)
            parsed_result = response_parse(llm_response)
            if parsed_result is None:
                logger.warning(f"Skipping case {input_identifier} due to LLM response parsing failure. Response was: '{llm_response[:100]}...'")
                return  # Exit early without further processing.

            # Only record results that were parsed successfully.
            async with self._lock:
                self.metrics.add_result(input_identifier, gt, ad, prob, reason, rprob, parsed_result)
        except Exception as e:
            logger.error(f"An unexpected error occurred while evaluating case {input_identifier}: {e}", exc_info=True)

    # Evaluate a batch asynchronously.
    async def evaluate(self, inputs: list[EvaluationInput]):
        tasks = [self.evaluate_single(input_data) for input_data in inputs]
        # asyncio.gather runs all tasks concurrently.
        await asyncio.gather(*tasks)

    def summary(self):
        self.metrics.summary()
