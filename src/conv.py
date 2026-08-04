import asyncio
import logging
from agents import Agent, Runner
import math
import json
from typing import Any
from pydantic import Field
import re

from .tools.tool_request import PatientContext, add_usage_inplace, make_empty_usage_dict
from .mes_collect import EvaluationOutputCollector as OutputCollector

logger = logging.getLogger(__name__)

INVALID_TOOL_TAGS = {
    "<think>", 
    "</think>",
    "<tool_call>", 
    "</tool_call>", 
    "<arg_key>", 
    "</arg_key>",
    "<arg_value>", 
    "</arg_value>"
}

MODEL_TURN_TIMEOUT_SECONDS = 800
MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS = 5
MAX_FORCE_WRAPUP_TURNS = 5
MAX_INVALID_FINISH_RETRIES = 2
INVALID_FINISH_RETRY_PROMPT = (
    "Your previous finish call was invalid because the diagnosis and/or reasoning "
    "arguments were empty. Retry the finish tool with non-empty diagnosis and reasoning."
)


def _extract_reasoning_summary(result: Any) -> str | None:
    raw_responses = getattr(result, "raw_responses", None) or []
    if not raw_responses:
        return None

    raw = raw_responses[0]
    output_items = getattr(raw, "output", None) or []

    summaries: list[str] = []
    for item in output_items:
        if getattr(item, "type", None) != "reasoning":
            continue

        summary_items = getattr(item, "summary", None) or []
        for summary in summary_items:
            text = getattr(summary, "text", None)
            if text:
                summaries.append(text)

    if not summaries:
        return None
    return "\n".join(summaries).strip()


def _get_case_label(context: PatientContext) -> str:
    return str(
        getattr(context, "case_id", None)
        or getattr(context, "patient_hadm_id", None)
        or getattr(context, "patient_id", "unknown_case")
    )


def _preview_last_message(messages: list[dict[str, Any]], limit: int = 160) -> str:
    if not messages:
        return "<empty>"

    content = messages[-1].get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return content[:limit].replace("\n", " ")


def _sdk_usage_to_dict(result: Any) -> dict[str, int]:
    """Normalize SDK usage objects into a plain dict for JSON output and summation."""
    usage = getattr(result.context_wrapper, "usage", None)
    if usage is None:
        return make_empty_usage_dict()

    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "requests": int(getattr(usage, "requests", 0) or 0),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
    }


def _sum_usage_parts(*parts: dict[str, int]) -> dict[str, int]:
    total = make_empty_usage_dict()
    for part in parts:
        add_usage_inplace(total, part)
    return total


async def run_simulation(
    doctor: Agent,
    patient: Agent,
    primary_symptom: str,
    context: PatientContext, 
    completion_prompt: str,
    output_collector: Any = Field(default_factory=OutputCollector),
    max_steps: int = 10,
):
    """
    Simulate an alternating doctor-patient conversation with the OpenAI Agents SDK.
    The patient starts, the doctor responds, and roles alternate until the doctor calls finish or reaches the step limit.
    """

    runner = Runner()

    starter = (
        f"My {primary_symptom}" if primary_symptom.strip()
        else "Hi, I'm not feeling well but not sure how to describe it. Could you help me figure out what's wrong?"
    )

    speaker = doctor
    listener = patient
    doctor_turns = 0
    doctor_mes = [{"role": "user", "content": starter}]
    patient_mes = [{"role": "assistant", "content": starter}]

    finish_called = False
    force_wrapup = False
    force_wrapup_turns = 0
    malformed_tool_call_retries = 0
    invalid_finish_retries = 0
    doctor_usage = make_empty_usage_dict()
    patient_usage = make_empty_usage_dict()
    case_label = _get_case_label(context)

    output_collector.display_message(name=patient.name, message=starter)
    while not finish_called:
        # print('doctor_mes:', doctor_mes)
        # print('patient_mes:', patient_mes)
        # The current agent generates a response; context is required.
        if speaker.name.lower().startswith("doctor"):
            # Enter forced wrap-up mode when needed.
            if doctor_turns >= max_steps and not force_wrapup:
                # Add the system wrap-up prompt.
                doctor_mes.append({"role": "system", "content": completion_prompt})
                force_wrapup = True
            # On subsequent forced wrap-up turns, restrict the doctor to finish.
            elif force_wrapup and not finish_called:
                force_wrapup_turns += 1
                if force_wrapup_turns > MAX_FORCE_WRAPUP_TURNS:
                    raise RuntimeError(
                        f"Exceeded force-wrapup limit for case={case_label} after {force_wrapup_turns - 1} non-finish doctor turns"
                    )
                doctor.tools = [t for t in doctor.tools if t.name.lower() == "finish"]
                doctor.model_settings.tool_choice="required"

            current_messages = doctor_mes
            logger.info(
                "Calling runner.run for case=%s speaker=%s doctor_turn=%s force_wrapup=%s",
                case_label,
                speaker.name,
                doctor_turns,
                force_wrapup,
            )
            try:
                result = await asyncio.wait_for(
                    runner.run(input=doctor_mes, starting_agent=speaker, context=context, max_turns=40),
                    timeout=MODEL_TURN_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                logger.error(
                    "Timed out waiting for model response: case=%s speaker=%s doctor_turn=%s last_message=%s",
                    case_label,
                    speaker.name,
                    doctor_turns,
                    _preview_last_message(current_messages),
                )
                raise RuntimeError(
                    f"Model turn timeout for case={case_label}, speaker={speaker.name}, doctor_turn={doctor_turns}"
                ) from exc
        else:
            current_messages = patient_mes
            logger.info(
                "Calling runner.run for case=%s speaker=%s doctor_turn=%s force_wrapup=%s",
                case_label,
                speaker.name,
                doctor_turns,
                force_wrapup,
            )
            try:
                result = await asyncio.wait_for(
                    runner.run(input=patient_mes, starting_agent=speaker, context=context, max_turns=20),
                    timeout=MODEL_TURN_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                logger.error(
                    "Timed out waiting for model response: case=%s speaker=%s doctor_turn=%s last_message=%s",
                    case_label,
                    speaker.name,
                    doctor_turns,
                    _preview_last_message(current_messages),
                )
                raise RuntimeError(
                    f"Model turn timeout for case={case_label}, speaker={speaker.name}, doctor_turn={doctor_turns}"
                ) from exc

        logger.info(
            "runner.run returned for case=%s speaker=%s doctor_turn=%s",
            case_label,
            speaker.name,
            doctor_turns,
        )

        # Each runner.run() call is one role-specific model turn, so we accumulate immediately.
        turn_usage = _sdk_usage_to_dict(result)
        if speaker.name.lower().startswith("doctor"):
            add_usage_inplace(doctor_usage, turn_usage)
        else:
            add_usage_inplace(patient_usage, turn_usage)

        # new_message = re.sub(r"<think>.*?<think>", "", result.final_output, flags=re.DOTALL).strip()
        new_message = result.final_output

        thinking_message = None
        if speaker.name.lower().startswith("doctor"):
            thinking_message = _extract_reasoning_summary(result)

        is_failed_tool_call = any(tag in new_message for tag in INVALID_TOOL_TAGS)
   
        if is_failed_tool_call:
            malformed_tool_call_retries += 1
            logger.warning(
                "Malformed tool call detected for case=%s speaker=%s retry=%s/%s output_preview=%s",
                case_label,
                speaker.name,
                malformed_tool_call_retries,
                MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS,
                (new_message or "")[:200].replace("\n", " "),
            )
            if malformed_tool_call_retries >= MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS:
                raise RuntimeError(
                    f"Exceeded malformed tool call retry limit for case={case_label}, speaker={speaker.name}"
                )
            continue
        malformed_tool_call_retries = 0

        new_input_items = [item.to_input_item() for item in result.new_items]
        # print('new_input_items:', new_input_items)

        # Store function calls until their outputs are available.
        tool_calls_pending_output = {} 
        invalid_finish_detected = False

        # First pass: analyze function calls immediately.
        for item in new_input_items:
            item_type = item.get("type")
            # --- Handle tool calls. ---
            if item_type in ["function_call", "function_call_output"]:
                # 1. By default, add the original item to the message history.
                item_for_history = item
                # doctor_mes.append(item)
                if item_type == "function_call":
                    # 2. For function calls, replace the default with a copy.
                    item_for_history = item.copy()
                    # 3. Safely remove logprobs from the copy when present.
                    if 'logprobs' in item_for_history:
                        del item_for_history['logprobs']

                    call_id = item.get("call_id")
                    function_name = item.get("name")
                    
                    # --- Analyze logprobs immediately for the finish tool. ---
                    if function_name == "finish":
                        finish_called = _analyze_and_record_finish_logprobs(item, output_collector)
                        if finish_called:
                            invalid_finish_retries = 0
                        else:
                            invalid_finish_detected = True
                            invalid_finish_retries += 1
                            logger.warning(
                                "Invalid finish call detected for case=%s retry=%s/%s arguments=%s",
                                case_label,
                                invalid_finish_retries,
                                MAX_INVALID_FINISH_RETRIES,
                                item.get("arguments"),
                            )
                            if invalid_finish_retries >= MAX_INVALID_FINISH_RETRIES:
                                raise RuntimeError(
                                    f"Exceeded invalid finish retry limit for case={case_label}"
                                )
                            doctor_mes.append({"role": "system", "content": INVALID_FINISH_RETRY_PROMPT})
                    
                    # Store every call while waiting for its output.
                    if call_id:
                        tool_calls_pending_output[call_id] = item
                # 3. Add the final item_for_history to doctor_mes.
                doctor_mes.append(item_for_history)

        actual_speaker_agent = speaker.name
        is_handoff_turn = False
        # Second pass: merge tool calls with outputs and record the final action.
        for item in new_input_items:
            item_type = item.get("type")

            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if call_id in tool_calls_pending_output:
                    call_item = tool_calls_pending_output[call_id]
                    
                    # Extract the information required for merging.
                    function_name = call_item.get("name")
                    arguments = call_item.get("arguments")
                    output = item.get("output")

                    # Record the merged action with the collector.
                    output_collector.display_action(
                        message=output,
                        function_name=function_name,
                        arguments=arguments
                    )

                    try:
                        # 1. Try to parse the output string as a Python object.
                        parsed_output = json.loads(output)
                        
                        # 2. Confirm the parsed object is a dictionary with an "assistant" key.
                        if isinstance(parsed_output, dict) and "assistant" in parsed_output:
                            is_handoff_turn = True                            
                            # 3. The value can now be read safely from the dictionary.
                            handoff_agent_name = parsed_output["assistant"]                            
                            print(f"INFO: Handoff to '{handoff_agent_name}' was handled.")
                            actual_speaker_agent = handoff_agent_name                           
                    except (json.JSONDecodeError, TypeError):
                        pass
                    
                    # Remove the completed call from the pending dictionary.
                    del tool_calls_pending_output[call_id]
        
        # Handle tool calls that did not receive an output.
        for call_id, call_item in tool_calls_pending_output.items():
            # Record only before the simulation ends, because finish output may arrive in the next loop.
            if not finish_called:
                 output_collector.display_action(
                    message="Tool call occurred, but no output was received in this turn.",
                    function_name=call_item.get("name"),
                    arguments=call_item.get("arguments")
                )

        if finish_called:
            break

        if invalid_finish_detected:
            if speaker.name.lower().startswith("doctor"):
                doctor_turns += 1
            speaker = doctor
            listener = patient
            continue

        # Record text messages.
        if new_message:
            # --- Decide how to handle output based on whether this is a handoff turn. ---
            if is_handoff_turn and actual_speaker_agent.lower() not in ["doctor", "patient"]:
                # Intercept and redirect messages from internal specialists.
                print(f"INFO: Intercepting internal message from {actual_speaker_agent}.")
                
                memo_for_doctor = (
                    f"SYSTEM MEMO FROM {actual_speaker_agent}:\n"
                    f"{new_message}"
                )
                
                # Update the doctor's message history directly.
                doctor_mes.append({"role": "system", "content": memo_for_doctor})
                output_collector.display_message(name=actual_speaker_agent, message=new_message, thinking=thinking_message)
                # Do not add the message to the patient's history.
            else:
                output_collector.display_message(name=speaker.name, message=new_message, thinking=thinking_message)
                if speaker.name.lower().startswith("doctor"):
                    doctor_mes.append({"role": "assistant", "content": new_message})
                    patient_mes.append({"role": "user", "content": new_message})
                else:
                    doctor_mes.append({"role": "user", "content": new_message})
                    patient_mes.append({"role": "assistant", "content": new_message})

        # Increment the turn count when the current agent is the doctor.
        if speaker.name.lower().startswith("doctor"):
            doctor_turns += 1

        # During forced wrap-up, allow only the doctor to respond.
        if force_wrapup:
            speaker = doctor
            listener = patient
        else:
            if is_handoff_turn and actual_speaker_agent.lower() not in ["doctor", "patient"]:
                # After a specialist message, the next turn must be the doctor.
                speaker = doctor
                listener = patient
                print("INFO: Control handed back to Doctor after specialist turn.")
            else:
                # Normal role alternation.
                speaker, listener = listener, speaker

    return {
        "doctor": doctor_usage,
        "patient": patient_usage,
        "total": _sum_usage_parts(doctor_usage, patient_usage),
    }



def _analyze_and_record_finish_logprobs(item: dict, output_collector: OutputCollector):
    """Analyze and record logprobs for the finish tool."""
    try:
        arguments_str = item.get("arguments", "{}")
        arguments_dict = json.loads(arguments_str)
        diagnosis_value = arguments_dict.get("diagnosis")
        reasoning_value = arguments_dict.get("reasoning")
        logprobs_list = item.get("logprobs", [])
        diagnosis_text = diagnosis_value.strip() if isinstance(diagnosis_value, str) else ""
        reasoning_text = reasoning_value.strip() if isinstance(reasoning_value, str) else ""

        if not diagnosis_text or not reasoning_text:
            print(
                "'finish' tool call did not contain non-empty 'diagnosis' and 'reasoning' arguments."
            )
            return False

        # Helper: greedily match a target text against the logprob sequence.
        def _match_argument_tokens(target_text: str, logprobs: list):
            """Return matched token dictionaries, or an empty list when matching fails."""
            if not target_text or not logprobs:
                return []

            search_phrase = target_text.lower()
            argument_logprobs = []

            temp_phrase = ""
            for lp in logprobs:
                token_text = lp.get("token", "")
                temp_phrase += token_text

                if search_phrase.startswith(temp_phrase.lower()):
                    argument_logprobs.append(lp)
                    if temp_phrase.lower() == search_phrase:
                        break
                else:
                    temp_phrase = ""
                    argument_logprobs = []
                    # Check whether the current token can start a new match.
                    if search_phrase.startswith(token_text.lower()):
                        temp_phrase = token_text
                        argument_logprobs.append(lp)

            # Treat an incomplete match of search_phrase as a failure.
            joined = "".join(lp.get("token", "") for lp in argument_logprobs).lower()
            if joined != search_phrase:
                return []
            return argument_logprobs

        # Helper: calculate sum, average, and geometric mean from token logprobs.
        def _compute_stats(argument_logprobs: list):
            total_logprob = sum(lp["logprob"] for lp in argument_logprobs)
            avg_logprob = total_logprob / len(argument_logprobs)
            geometric_mean_prob = math.exp(avg_logprob)
            return {
                "sum_of_logprobs": total_logprob,
                "avg_logprob": avg_logprob,
                "geometric_mean_probability": geometric_mean_prob,
                "tokens": [
                    {"token": lp["token"], "logprob": lp["logprob"]}
                    for lp in argument_logprobs
                ],
            }
                

        if logprobs_list:
            analysis_data_to_save = {}

            # ------- 1) Diagnosis section (preserve existing behavior). -------
            if diagnosis_value:
                diag_argument_logprobs = _match_argument_tokens(
                    diagnosis_value, logprobs_list
                )
                if diag_argument_logprobs:
                    analysis_data_to_save["diagnosis"] = _compute_stats(
                        diag_argument_logprobs
                    )
                else:
                    print(
                        f"Could not find a matching sequence of logprobs for diagnosis '{diagnosis_value}'."
                    )

            # ------- 2) Reasoning section (new). -------
            if reasoning_value:
                reasoning_argument_logprobs = _match_argument_tokens(
                    reasoning_value, logprobs_list
                )
                if reasoning_argument_logprobs:
                    analysis_data_to_save["reasoning"] = _compute_stats(
                        reasoning_argument_logprobs
                    )
                else:
                    print(
                        f"Could not find a matching sequence of logprobs for reasoning '{reasoning_value}'."
                    )

            # Record the analysis when at least one argument matched.
            if analysis_data_to_save:
                output_collector.display_logprob_analysis(
                    function_name="finish",
                    arguments=arguments_dict,
                    analysis_results=analysis_data_to_save,
                )
            else:
                print(
                    "No matching token sequences found for either diagnosis or reasoning."
                )
        else:
            print("'finish' tool call did not contain 'logprobs'. Skipping logprob analysis.")
        return True
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"An error occurred during logprob analysis: {e}")
        return False
