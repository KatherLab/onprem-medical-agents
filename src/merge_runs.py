import glob
import json
import math
import os
from collections import defaultdict
from typing import Any

TRACE_TOOL_ARGUMENTS = {
    "request_physical_exam": [],
    "request_blood_test": ["lab_values", "test_names", "requested_tests"],
    "request_urine_test": ["urine_values", "test_names", "requested_tests"],
    "request_microbiology": ["microbiology_tests", "test_names", "requested_tests"],
    "request_bedside_test": ["test_names", "requested_tests"],
    "request_other_investigation": ["test_names", "requested_tests"],
    "request_radiology": ["modality", "region"],
}


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    """Normalize stored action arguments into a dictionary when possible."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_trace_atom(value: Any) -> str:
    """Canonical string normalization used for tool-trace payloads."""
    return " ".join(str(value).strip().lower().split())


def _normalize_trace_value(value: Any) -> str:
    """Normalize one tool argument while ignoring order inside list-valued requests."""
    if isinstance(value, list):
        normalized_items = sorted(
            {
                _normalize_trace_atom(item)
                for item in value
                if str(item).strip()
            }
        )
        return "|".join(normalized_items)

    if isinstance(value, dict):
        parts = []
        for key in sorted(value):
            normalized = _normalize_trace_value(value[key])
            if normalized:
                parts.append(f"{_normalize_trace_atom(key)}={normalized}")
        return ";".join(parts)

    return _normalize_trace_atom(value)


def _normalize_tool_invocation(tool_name: str, arguments: Any) -> str | None:
    """Convert one diagnostic tool call into a stable, order-preserving trace token."""
    if tool_name not in TRACE_TOOL_ARGUMENTS:
        return None

    parsed_arguments = _parse_arguments(arguments)
    keys = TRACE_TOOL_ARGUMENTS[tool_name]
    if not keys:
        return tool_name

    payload_parts = []
    for key in keys:
        normalized_value = _normalize_trace_value(parsed_arguments.get(key))
        if normalized_value:
            payload_parts.append(f"{key}={normalized_value}")

    if not payload_parts:
        return tool_name

    return f"{tool_name}::" + ";".join(payload_parts)


def _extract_diagnostic_tool_trace(conversation_path: str) -> list[str]:
    """Extract an ordered sequence of normalized diagnostic tool invocations."""
    try:
        with open(conversation_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []

    trace: list[str] = []
    for message in messages:
        if message.get("type") != "action":
            continue
        tool_name = message.get("function")
        normalized_invocation = _normalize_tool_invocation(tool_name, message.get("arguments"))
        if normalized_invocation:
            trace.append(normalized_invocation)

    return trace

def _find_conversation_file(
    conversation_search_root: str,
    run_id: int,
    dataset: str,
    hadm_id: Any,
) -> str | None:
    """
    Locate the stored conversation JSON for one run/case pair.
    """
    pattern = os.path.join(
        conversation_search_root,
        "**",
        f"run_{run_id}",
        dataset,
        f"{dataset}_{hadm_id}_conversation_*.json",
    )
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None

    return sorted(candidates)[0]


def _compute_bottomk_probability(tokens: list[dict[str, Any]], k: int = 3) -> float | None:
    """
    Compute the mean probability of the lowest-probability diagnosis tokens.

    For chosen-token logprobs ``l_i = log(p_i)``, the run-level score is:
    ``mean(sorted(exp(l_i))[:min(k, n_tokens)])``.
    """
    probabilities: list[float] = []
    for token in tokens:
        logprob = token.get("logprob")
        if logprob is None:
            continue
        try:
            probabilities.append(float(math.exp(float(logprob))))
        except (TypeError, ValueError, OverflowError):
            continue

    if not probabilities:
        return None

    bottom_k = sorted(probabilities)[: min(k, len(probabilities))]
    return float(sum(bottom_k) / len(bottom_k))


def _extract_bottom3_probability(conversation_path: str) -> float | None:
    """
    Read the saved finish logprob analysis and derive ``Bottom3Prob_Dx`` post hoc.
    """
    try:
        with open(conversation_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    for message in reversed(messages):
        if message.get("type") != "logprob_analysis" or message.get("function") != "finish":
            continue
        analysis = message.get("analysis", {})
        diagnosis = analysis.get("diagnosis", {}) if isinstance(analysis, dict) else {}
        tokens = diagnosis.get("tokens", []) if isinstance(diagnosis, dict) else []
        if isinstance(tokens, list):
            return _compute_bottomk_probability(tokens, k=3)
        break

    return None


def collect_and_aggregate_runs(
    logs_base_dir: str,
    num_runs: int,
    dataset: str,
    conversation_search_root: str,
):
    """
    Collect and aggregate N independent runs for a dataset into one file.
    """
    # defaultdict makes it convenient to append run data to lists.
    aggregated_data = defaultdict(lambda: {'runs': []})

    print(f"Aggregating {num_runs} runs for dataset: {dataset}...")

    for i in range(1, num_runs + 1):
        # --- Core path construction. ---
        # Build the evaluation-result path for each run dynamically.
        # Layout: /path/to/logs/run_{i}/dataset_name/dataset_name_diagnosis_results.jsonl
        input_file = os.path.join(
            logs_base_dir, 
            f"run_{i}", 
            dataset, 
            f"{dataset}_diagnosis_results.jsonl"
        )
        # --- End path construction. ---

        if not os.path.exists(input_file):
            print(f"  警告: Run {i} 的评估文件未找到，已跳过: {input_file}")
            continue

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    hadm_id_raw = data.get('hadm_id')
                    if hadm_id_raw is None:
                        continue
                    hadm_id = str(hadm_id_raw).strip()
                    if not hadm_id:
                        continue
                    
                    # Populate ground_truth, assuming it is identical across runs.
                    if 'ground_truth' not in aggregated_data[hadm_id]:
                        aggregated_data[hadm_id]['ground_truth'] = data.get('ground_truth')
                    
                    # Collect the core information for each run.
                    # Important: ensure evaluate.py outputs all of these fields.
                    decision = bool(data.get("decision", False))
                    partial_credit = bool(data.get("partial_credit", False))
                    effective_decision = decision or partial_credit
                    conversation_path = _find_conversation_file(
                        conversation_search_root=conversation_search_root,
                        run_id=i,
                        dataset=dataset,
                        hadm_id=hadm_id,
                    )
                    diagnosis_bottom3_prob = (
                        _extract_bottom3_probability(conversation_path)
                        if conversation_path
                        else None
                    )
                    diagnostic_tool_trace = (
                        _extract_diagnostic_tool_trace(conversation_path)
                        if conversation_path
                        else []
                    )

                    run_info = {
                        'run_id': i,
                        'assistant_diagnosis': data.get('assistant_diagnosis'),
                        'diagnosis_prob': data.get('geometric_mean_probability'), 
                        # Bottom3Prob_Dx is computed post hoc from saved diagnosis token logprobs.
                        'diagnosis_bottom3_prob': diagnosis_bottom3_prob,
                        'diagnostic_tool_trace': diagnostic_tool_trace,
                        'decision': decision,
                        'partial_credit': partial_credit,
                        'effective_decision': effective_decision,
                        # 'reported_confidence': data.get('reported_confidence_percentage')
                        'assistant_reasoning': data.get('assistant_reasoning'),
                        'reason_prob': data.get('reason_geometric_mean_probability')
                    }
                    aggregated_data[hadm_id]['runs'].append(run_info)
                except json.JSONDecodeError:
                    print(f"  警告: 在文件 {input_file} 中发现无效的JSON行，已跳过。")
                    continue
    
    # --- Write the aggregated data to a new JSONL file. ---
    if not aggregated_data:
        print(f"没有找到任何可聚合的数据用于数据集: {dataset}。")
        return

    output_path = os.path.join(logs_base_dir, f"{dataset}_multi_run.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for hadm_id, data in aggregated_data.items():
            record = {
                'hadm_id': hadm_id,
                'ground_truth': data['ground_truth'],
                'runs': data['runs']
            }
            f.write(json.dumps(record) + '\n')
            
    print(f"聚合完成！共处理了 {len(aggregated_data)} 个独特的HADM_ID。")
    print(f"数据已保存到 {output_path}")

def main():
    logs_base_dir = '/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/evaluation_logs/63/10runs_63_104/63.10'
    # Point this at the concrete history root whose direct children are run_1, run_2, ...
    conversation_search_root = '/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/history/63_clean_clean_10runs_104'
    BENCHMARK = "AIDOC" #"VIVABENCH" 
    if BENCHMARK == "VIVABENCH":
        ALL_DATASETS = [
        "vivabench_pubmed",
        ]
    else:
        ALL_DATASETS = [
            "appendicitis",
            "cholecystitis", 
            "diverticulitis", 
            "pancreatitis", 
            "pneumonia", 
            "pulmonary_embolism", 
            "uti"
        ]
    for dataset in ALL_DATASETS:
        collect_and_aggregate_runs(
            logs_base_dir=logs_base_dir,
            num_runs=10,  ###### change here
            dataset=dataset,
            conversation_search_root=conversation_search_root,
        )

if __name__ == "__main__":
    main()
