import os
import re
import glob
import logging
import asyncio
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from openai import AsyncOpenAI
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.agent import StopAtTools
from agents.models.interface import Model
from agents.extensions.models.litellm_model import LitellmModel
from tenacity import retry, stop_after_attempt, wait_fixed, AsyncRetrying
from tqdm import tqdm
from termcolor import colored
import argparse
from typing import Any

# ========== Project modules ========== #
from .configs.agent_config import *
from .logging_config import setup_logging
from .conv import run_simulation
from .mes_collect import EvaluationOutputCollector as OutputCollector
from .dataset.mimic_dataset import MIMIC_Dataset
from .dataset.vivabench_dataset import VivaBenchCase, VivaBenchDataset
from .tools import TOOLS
from .tools.tool_request import PatientContext, add_usage_inplace, make_empty_usage_dict
from .multi_agents import critic_handoff,executor_handoff

if BENCHMARK == "VIVABENCH":
    from .prompts_vivabench import (
        VIVABENCH_MEDICAL_SYSTEM_PROMPT as MEDICAL_SYSTEM_PROMPT,
        VIVABENCH_PATIENT_SYSTEM_PROMPT as PATIENT_SYSTEM_PROMPT,
        COMPLETION_PROMPT,
        CRITIC_SYSTEM_PROMPT,
    )
else:
    from .prompts import *


# ========== Logging configuration ========== #
setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)

set_tracing_disabled(True)

# ========== Configuration data class ========== #
@dataclass
class SimulationConfig:
    save_dir: str
    ds_name: str
    ds: Any
    medical_assistant_model: str
    medical_assistant_temperature: float
    patient_assistant_model: str
    patient_assistant_temperature: float
    medical_system_prompt: str
    critic_system_prompt: str
    completion_prompt: str
    patient_system_prompt: str
    max_samples: Optional[int] = None
    max_workers: int = 4
    selected_hadm_ids: Optional[List[Any]] = None


def build_agent_model(model_name: str, api_key: str | None, base_url: str | None) -> Model:
    if model_name.startswith("litellm/"):
        return LitellmModel(
            model=model_name.removeprefix("litellm/"),
            api_key=api_key,
            base_url=base_url,
        )

    normalized_model_name = model_name.removeprefix("openai/")
    return OpenAIChatCompletionsModel(
        model=normalized_model_name,
        openai_client=AsyncOpenAI(api_key=api_key, base_url=base_url),
    )


# ========== Core class ========== #
class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.patient_gen = self.patient_iterator()

    def patient_iterator(self):
        ds = self.config.ds
        candidate_ids = get_dataset_case_ids(ds)
        if self.config.selected_hadm_ids is None:
            case_ids = list(candidate_ids)
        else:
            # An empty selection means there are no remaining cases to run.
            case_ids = [cid for cid in candidate_ids if cid in self.config.selected_hadm_ids]
        if self.config.max_samples:
            case_ids = case_ids[:self.config.max_samples]

        logger.info(colored(f"🧬 Preparing {len(case_ids)} simulations for {self.config.ds_name} ...", "red"))
        for case_id in case_ids:
            yield self.prepare_patient(case_id)

    def prepare_patient(self, case_id: str | int):
        patient_data = self.config.ds[case_id]
        primary_symptom = get_case_primary_symptom(patient_data)
        final_anamnesis = build_case_anamnesis(patient_data)
        external_case_id = get_case_identifier(patient_data)
        subject_id = get_subject_identifier(patient_data)

        collector = OutputCollector(
            hadm_id=external_case_id,
            dataset_name=self.config.ds_name,
            save_dir=self.config.save_dir,
        )

        context = PatientContext(
            patient_id=subject_id,
            patient_hadm_id=external_case_id,
            patient_data=patient_data,
            benchmark=BENCHMARK,
            case_id=external_case_id,
        )

        med_assistant = Agent[PatientContext](
            name="Doctor",
            instructions=self.config.medical_system_prompt,
            tools=TOOLS,
            handoffs=[critic_handoff],
            tool_use_behavior=StopAtTools(stop_at_tool_names=["finish"]),
            model=build_agent_model(
                self.config.medical_assistant_model,
                MEDICAL_ASSISTANT_API_KEY,
                MEDICAL_ASSISTANT_BASE_URL,
            ),
            model_settings=ModelSettings(
                temperature=self.config.medical_assistant_temperature,
                tool_choice="auto", #"required",
                # response_include=["message.output_text.logprobs"], # for openai models, only for message, not for tool
                extra_args={
                    'logprobs': True,
                    }, # for local models
                # parallel_tool_calls=False,
                max_tokens=24576, #16384, #8192
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    }, 
            ),
        )

        patient_assistant = Agent[PatientContext](
            name="Patient",
            instructions=self.config.patient_system_prompt.format(
                primary_symptom=primary_symptom, anamnesis_summary=final_anamnesis),
            model=build_agent_model(
                self.config.patient_assistant_model,
                PATIENT_ASSISTANT_API_KEY,
                PATIENT_ASSISTANT_BASE_URL,
            ),
            model_settings=ModelSettings(temperature=self.config.patient_assistant_temperature, max_tokens=8192),
        )

        return med_assistant, patient_assistant, primary_symptom, context, collector

    # Run simulations asynchronously.
    async def run_async(self):
        logger.info(f"🚀 Using asyncio for concurrency")

        # Use an asyncio semaphore to control concurrency.
        semaphore = asyncio.Semaphore(self.config.max_workers)

        tasks = []
        for data in self.patient_gen:
            task = asyncio.create_task(self.run_patient_task_with_limiter(data, semaphore))
            tasks.append(task)

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Running Simulations"):
            hadm_id, error = await future
            print('hadm_id:', hadm_id)
            if error:
                logger.error(f"❌ Failed HADM_ID {hadm_id}: {error}")
            else:
                logger.info(f"✅ Completed HADM_ID {hadm_id}")

    async def run_patient_task_with_limiter(self, data, semaphore):
        hadm_id = get_hadm_id_safe(data)
        async with semaphore:
            try:
                # Use Tenacity for asynchronous retries.
                async for attempt in AsyncRetrying(stop=stop_after_attempt(3), wait=wait_fixed(1)):
                    with attempt:
                        await self.run_patient_task(data)
                return hadm_id, None
            except Exception as e:
                return hadm_id, e

    async def run_patient_task(self, data):
        medical_assistant, patient_assistant, primary_symptom, context, collector = data
        case_start_time = datetime.now().isoformat()
        start_perf = time.perf_counter()

        conversation_usage = await run_simulation(
            medical_assistant, 
            patient_assistant, 
            primary_symptom, 
            context, 
            output_collector=collector,
            completion_prompt=self.config.completion_prompt)

        case_end_time = datetime.now().isoformat()
        wall_clock_seconds = time.perf_counter() - start_perf

        total_usage = make_empty_usage_dict()
        add_usage_inplace(total_usage, conversation_usage.get("doctor"))
        add_usage_inplace(total_usage, conversation_usage.get("patient"))
        add_usage_inplace(total_usage, context.blood_matcher_usage)

        collector.display_run_metadata(
            {
                "case_start_time": case_start_time,
                "case_end_time": case_end_time,
                "wall_clock_seconds": round(wall_clock_seconds, 6),
                "token_usage": {
                    "doctor": conversation_usage.get("doctor", make_empty_usage_dict()),
                    "patient": conversation_usage.get("patient", make_empty_usage_dict()),
                    "blood_matcher": context.blood_matcher_usage,
                    "total": total_usage,
                },
                "deployment": {
                    "doctor_model": self.config.medical_assistant_model,
                    "patient_model": self.config.patient_assistant_model,
                    "blood_matcher_model": (
                        VIVABENCH_BLOOD_MATCHER_MODEL if BENCHMARK == "VIVABENCH" else None
                    ),
                },
            }
        )


# ========== Helper functions ========== #
def get_admission_medication(patient):
    try:
        meds = patient.history_pe_admedication_diagnosis["admission_medication"]
        return meds.values[0] if len(meds.values) > 0 and meds.values[0] else "No current medication."
    except Exception:
        return "No current medication."


def get_admission_chief_complaint(patient):
    if BENCHMARK == "AIDOC":
        cc = patient.triage.chiefcomplaint
        return f"primary symptom: {cc.values[0]}" if len(cc.values) > 0 else ""
    elif BENCHMARK == "CDM":
        return "primary symptom: abdominal pain."


def get_dataset_case_ids(ds: Any) -> list[Any]:
    """Provide one iteration interface for both MIMIC and VivaBench datasets."""
    if hasattr(ds, "hadm_ids"):
        return list(ds.hadm_ids)
    if hasattr(ds, "case_ids"):
        return list(ds.case_ids)
    raise AttributeError("Dataset must expose either `hadm_ids` or `case_ids`.")


def get_case_identifier(patient: Any) -> str:
    if hasattr(patient, "case_id"):
        return str(patient.case_id)
    return str(patient.admissions.hadm_id.values[0])


def get_subject_identifier(patient: Any) -> str:
    if hasattr(patient, "case_id"):
        return str(patient.case_id)
    return str(patient.patients.subject_id.values[0])


def get_case_primary_symptom(patient: Any) -> str:
    if hasattr(patient, "chief_complaint"):
        complaint = patient.chief_complaint
        return f"primary symptom: {complaint}" if complaint else ""
    return get_admission_chief_complaint(patient)


def build_case_anamnesis(patient: Any) -> str:
    """Build the prompt summary from benchmark-specific case fields."""
    if hasattr(patient, "history_freetext"):
        sections: list[str] = []
        demographics = patient.demographics
        history = patient.history

        if demographics:
            age = demographics.get("age")
            gender = demographics.get("gender")
            age_unit = demographics.get("unit", "")
            demo_bits = [str(x) for x in [age, age_unit, gender] if x]
            if demo_bits:
                sections.append("Demographics: " + " ".join(demo_bits))

        chief_complaint = history.get("chief_complaint")
        if chief_complaint:
            sections.append(f"Chief complaint: {chief_complaint}")

        symptoms = history.get("symptoms", {})
        symptom_notes = []
        for symptom_name, payload in symptoms.items():
            if payload.get("present"):
                description = payload.get("history") or payload.get("character") or payload.get("name") or symptom_name
                symptom_notes.append(f"- {description}")
        if symptom_notes:
            sections.append("Present symptoms:\n" + "\n".join(symptom_notes[:10]))

        social_history = history.get("social_history", {})
        if social_history:
            sections.append(f"Social history: {social_history}")

        if patient.history_freetext:
            sections.append(f"History summary: {patient.history_freetext}")

        if patient.medications:
            sections.append(f"Medications: {patient.medications}")
        if patient.allergies:
            sections.append(f"Allergies: {patient.allergies}")
        return "\n\n".join([s for s in sections if s])

    base_anamnesis = patient.history_pe_admedication_diagnosis["extracted_history"].values[0]
    if BENCHMARK == "AIDOC":
        medication_str = get_admission_medication(patient)
        if medication_str and medication_str.strip():
            medication_header = "\n\nMedication:\n"
            return base_anamnesis + medication_header + medication_str
    return base_anamnesis


def get_hadm_id_safe(data_tuple: Tuple) -> Optional[str]:
    try:
        context = data_tuple[3]
        return context.case_id or context.patient_hadm_id
    except Exception:
        return None


# ========== Main entry point ========== #
if __name__ == "__main__":
    # --- Step 1: Add the command-line argument parser. ---
    parser = argparse.ArgumentParser(description="Run a medical simulation batch with a specific run ID.")
    parser.add_argument('--run_id', type=int, required=True, help='A unique integer ID for this simulation run (e.g., 1, 2, 3...).')
    parser.add_argument('--case_id', type=str, default=None, help='Optional benchmark case identifier, mainly for running a single VivaBench case.')
    args = parser.parse_args()
    
    # Read run_id from the parsed arguments.
    run_id = args.run_id
    requested_case_id = args.case_id

    async def main():
        # --- Step 2: Build the root output directory from run_id. ---
        # Base output directory without run_id.
        base_output_dir = FORMATTED_OUTPUT_ROOT
        
        # Root output directory for this run.
        # For example, run_id=1 produces .../formatted_outputs/run_1.
        run_specific_base_dir = os.path.join(base_output_dir, f"run_{run_id}")
        
        print(f"==============================================")
        print(f"STARTING SIMULATION BATCH | RUN ID: {run_id}")
        print(f"Output will be saved under: {run_specific_base_dir}")
        print(f"==============================================")

        runtime_dataset_names = DATASET_NAMES
        if BENCHMARK == "VIVABENCH":
            runtime_dataset_names = ["vivabench_pubmed"]

        for dataset_name in runtime_dataset_names:
            if BENCHMARK == "VIVABENCH":
                ds = VivaBenchDataset.load_jsonl(VIVABENCH_DATA_PATH)
                original_ids = ds.case_ids
                if requested_case_id is not None:
                    if requested_case_id not in ds.case_ids:
                        raise ValueError(f"Requested case_id '{requested_case_id}' was not found in VivaBench dataset.")
                    original_ids = [requested_case_id]
            elif BENCHMARK == "CDM":
                original_ids = CDM_HADM_IDS[dataset_name]
                ds = MIMIC_Dataset.load_from_pkl(dataset_name)
            else:
                original_ids = AIDOC_HADM_IDS[dataset_name]
                ds = MIMIC_Dataset.load_dataset(dataset_name)

            # --- Step 3: Create a dataset subdirectory for this run. ---
            # For example: .../formatted_outputs/run_1/pneumonia.
            output_dir_for_dataset = os.path.join(run_specific_base_dir, dataset_name)
            os.makedirs(output_dir_for_dataset, exist_ok=True)

            existing_jsons = glob.glob(os.path.join(output_dir_for_dataset, f"{dataset_name}_*_conversation_*.json"))
            if BENCHMARK == "VIVABENCH":
                pattern = re.compile(rf"{dataset_name}_(.+?)_conversation_\d+\.json")
                processed_ids = {
                    match.group(1) for f in existing_jsons if (match := pattern.match(os.path.basename(f)))
                }
            else:
                pattern = re.compile(rf"{dataset_name}_(\d+)_conversation_\d+\.json")
                processed_ids = {
                    int(match.group(1)) for f in existing_jsons if (match := pattern.match(os.path.basename(f)))
                }

            remaining_ids = [case_identifier for case_identifier in original_ids if case_identifier not in processed_ids]
            # Report progress with the logger or print.
            log_message = f"🔍 {dataset_name} (Run ID: {run_id}) | Total: {len(original_ids)} | Done: {len(processed_ids)} | Remaining: {len(remaining_ids)}"
            logger.info(log_message)  # Use the logger for progress reporting.

            config = SimulationConfig(
                save_dir=output_dir_for_dataset,
                ds_name=dataset_name,
                ds=ds,
                medical_assistant_model=MEDICAL_ASSISTANT_MODEL,
                medical_assistant_temperature=MEDICAL_ASSISTANT_TEMPERATURE,
                patient_assistant_model=PATIENT_ASSISTANT_MODEL,
                patient_assistant_temperature=PATIENT_ASSISTANT_TEMPERATURE,
                medical_system_prompt=MEDICAL_SYSTEM_PROMPT,
                critic_system_prompt=CRITIC_SYSTEM_PROMPT,
                completion_prompt=COMPLETION_PROMPT,
                patient_system_prompt=PATIENT_SYSTEM_PROMPT,
                max_samples=MAX_SAMPLES,
                max_workers=MAX_EVAL_PARALLEL_SIZE,
                selected_hadm_ids=remaining_ids,
            )

            await SimulationRunner(config).run_async()
    asyncio.run(main())
