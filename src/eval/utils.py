import json
import re
from typing import Any, Dict, Optional
import logging

from ..configs.agent_config import BENCHMARK

logger = logging.getLogger(__name__)


def response_parse(content: str) -> dict | None:
    """
    Parse LLM-generated content and extract JSON with normalized boolean values.

    Tolerates:
    - ```json ... ``` wrappers
    - leading / trailing non-JSON text
    - missing final closing brace in simple cases
    """
    if not content or not isinstance(content, str):
        logger.warning("Could not parse LLM response: empty or non-string content.")
        return None

    raw_content = content.strip()

    # 1. Prefer fenced JSON blocks if present
    matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
    if matches:
        candidate = matches[0].strip()
    else:
        candidate = raw_content

    # 2. If there is extra leading/trailing text, try to isolate the outermost JSON object
    first_brace = candidate.find("{")
    last_brace = candidate.rfind("}")
    if first_brace != -1:
        if last_brace != -1 and last_brace > first_brace:
            candidate = candidate[first_brace:last_brace + 1]
        else:
            # Missing final brace: try a conservative repair
            candidate = candidate[first_brace:] + "}"

    # 3. Normalize booleans
    candidate = re.sub(
        r"\b(?:true|false)\b",
        lambda m: m.group(0).lower(),
        candidate,
        flags=re.IGNORECASE,
    )

    # 4. Remove trailing commas before } or ]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            logger.warning(f"Could not parse LLM response: top-level JSON is not an object. Content: {candidate}")
            return None

        parsed = {k.lower(): v for k, v in parsed.items()}

        parsed["decision"] = str(parsed.get("decision", "")).lower() == "true"
        parsed["partial_credit"] = str(parsed.get("partial_credit", "")).lower() == "true"

        missed = parsed.get("missed_gold_diagnoses", [])
        parsed["missed_gold_diagnoses"] = missed if isinstance(missed, list) else []

        if parsed["decision"]:
            parsed["partial_credit"] = False

        reasoning = parsed.get("reasoning", "")
        parsed["reasoning"] = reasoning if isinstance(reasoning, str) else str(reasoning)

        return parsed

    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse LLM response: {e}. Content: {candidate}")
        return None



def get_assistant_diagnosis(conversation_path: str) -> Dict[str, Any]:
    """
    Locate the final 'finish' tool call in the conversation history and extract related information.
    """
    final_info = {
        "assistant_diagnosis": None,
        "assistant_reasoning": None,
        "geometric_mean_probability": None,
        "reason_geometric_mean_probability": None,
    }

    try:
        with open(conversation_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(
            f"[⚠️ERROR] Could not read or parse conversation file: {conversation_path}. Reason: {e}"
        )
        return final_info

    # Iterate backward because the finish call is usually last.
    for m in reversed(messages):
        # 1. Locate the primary finish tool-call event.
        if m.get("type") == "action" and m.get("function") == "finish":
            arguments = json.loads(m.get("arguments", "{}"))
            final_info["assistant_diagnosis"] = arguments.get("diagnosis")
            final_info["assistant_reasoning"] = arguments.get("reasoning")

        # 2. Locate probability information.
        elif m.get("type") == "logprob_analysis" and m.get("function") == "finish":
            analysis = m.get("analysis", {})

            # Initialize both variables to avoid UnboundLocalError.
            diag_prob = None
            reason_prob = None

            # Support the legacy structure, where analysis stores geometric_mean_probability at the top level.
            top_prob = analysis.get("geometric_mean_probability", None)
            if top_prob is not None:
                diag_prob = top_prob  # In the legacy structure, treat it as the diagnosis probability.

            else:
                # New structure: analysis = {"diagnosis": {...}, "reasoning": {...}}
                diag_stats = analysis.get("diagnosis")
                if isinstance(diag_stats, dict):
                    diag_prob = diag_stats.get("geometric_mean_probability", None)

                reason_stats = analysis.get("reasoning")
                if isinstance(reason_stats, dict):
                    reason_prob = reason_stats.get("geometric_mean_probability", None)

            if diag_prob is not None:
                try:
                    final_info["geometric_mean_probability"] = float(diag_prob)
                except (TypeError, ValueError):
                    pass

            if reason_prob is not None:
                try:
                    final_info["reason_geometric_mean_probability"] = float(reason_prob)
                except (TypeError, ValueError):
                    pass

    if final_info["assistant_diagnosis"] is None:
        print(
            f"[⚠️WARNING] No valid 'finish' tool call found in: {conversation_path}"
        )

    return final_info




def get_ground_truth_diagnosis(patient_data, patho: Optional[str] = None) -> Optional[str]:
    """
    patho: The matching criterion for the CDM benchmark, such as 'appendicitis'.
    """
    icd = patient_data.diagnosis_icd
    if BENCHMARK=='AIDOC':
        valid = icd[icd["is_invalid"] == False]
    elif BENCHMARK=='CDM':
        valid = icd

    if valid.empty:
        print(f"[⚠️WARNING] ground truth diagnosis missing | hadm_id: {patient_data.hadm_id}")
        return None
    
    # For CDM, prefer the long_title that contains patho.
    if BENCHMARK == 'CDM' and patho:
        key = patho.lower()
        for _, row in valid.iterrows():
            title = str(row.get("long_title", ""))
            if key in title.lower():
                return title
            
    return valid.iloc[0]["long_title"]


def save_decision_differences(model1_path: str, model2_path: str, output_path: str):
    """
    Compare decision-field differences between two model diagnosis-evaluation files and save them.

    Args:
        model1_path (str): Path to the first model result file.
        model2_path (str): Path to the second model result file.
        output_path (str): Path for the saved differences.
    """

    def load_jsonl(path):
        results = {}
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                results[item["hadm_id"]] = item
        return results

    # Load model results.
    model1 = load_jsonl(model1_path)
    model2 = load_jsonl(model2_path)

    # Compare results and collect differences.
    diffs = []
    for hadm_id in model1:
        if hadm_id in model2 and model1[hadm_id]["decision"] != model2[hadm_id]["decision"]:
            diffs.append({
                "hadm_id": hadm_id,
                "ground_truth": model1[hadm_id]["ground_truth"],
                "assistant_diagnosis": model1[hadm_id]["assistant_diagnosis"],
                "model1_decision": model1[hadm_id]["decision"],
                "model1_reasoning": model1[hadm_id].get("reasoning", ""),
                "model2_decision": model2[hadm_id]["decision"],               
                "model2_reasoning": model2[hadm_id].get("reasoning", "")
            })

    # # Save as JSONL.
    # with open(output_path, "w", encoding="utf-8") as f:
    #     for entry in diffs:
    #         f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # Save as a standard JSON array.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diffs, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(diffs)} decision differences to: {output_path}")




if __name__ == '__main__':
    save_decision_differences(
        model1_path="/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/evaluation_logs/cholecystitis/4.1_cholecystitis_diagnosis_Llama-3.3-70B-Instruct_78.jsonl",
        model2_path="/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/evaluation_logs/cholecystitis/4.2_cholecystitis_diagnosis_81.33.jsonl",
        output_path="/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/evaluation_logs/cholecystitis/cholecystitis_diagnosis_decision_diff_4.1_4.2.json"
    )
