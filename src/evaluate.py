import argparse
import asyncio
import glob
import json
import logging
import os
import re

from openai import AsyncOpenAI
from tqdm import tqdm

from .configs.agent_config import *
from .dataset.mimic_dataset import MIMIC_Dataset
from .dataset.vivabench_dataset import VivaBenchDataset
from .eval.evaluation_input import EvaluationInput
from .eval.evaluation_input_vivabench import VivaBenchEvaluationInput
from .eval.evaluators import DiagnosisEvaluator
from .logging_config import setup_logging

setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)


DATASET_TO_CATEGORY = {
    "appendicitis": "Appendicitis",
    "cholecystitis": "Cholecystitis",
    "diverticulitis": "Diverticulitis",
    "pancreatitis": "Pancreatitis",
    "pneumonia": "Pneumonia",
    "pulmonary_embolism": "Pulmonary Embolism",
    "uti": "Urinary Tract Infections",
}
VIVABENCH_DATASET_NAME = "vivabench_pubmed"


def _default_output_root() -> str:
    return "/mnt/bulk-uranus/lizhang/LiWorkSpace/data/evaluation/96/96.10"


def _default_conv_root() -> str:
    return "/mnt/bulk-uranus/lizhang/LiWorkSpace/data/history/96_clean"


def _resolve_dataset_list(dataset_arg: str) -> list[str]:
    if BENCHMARK == "VIVABENCH":
        if dataset_arg.lower() not in {"all", VIVABENCH_DATASET_NAME}:
            logger.warning(
                "Ignoring dataset=%s for VivaBench; using %s.",
                dataset_arg,
                VIVABENCH_DATASET_NAME,
            )
        return [VIVABENCH_DATASET_NAME]

    return list(DATASET_TO_CATEGORY.keys()) if dataset_arg.lower() == "all" else [dataset_arg]


def _resolve_category(dataset: str) -> str:
    if BENCHMARK == "VIVABENCH":
        return "VivaBench"
    return DATASET_TO_CATEGORY[dataset]


def _load_eval_dataset(dataset: str):
    if BENCHMARK == "VIVABENCH":
        return VivaBenchDataset.load_jsonl(VIVABENCH_DATA_PATH)
    if BENCHMARK == "CDM":
        return MIMIC_Dataset.load_from_pkl(dataset)
    return MIMIC_Dataset.load_dataset(dataset)


def _find_conversation_files(current_conv_dir: str, dataset: str) -> list[str]:
    return glob.glob(os.path.join(current_conv_dir, f"{dataset}_*_conversation_*.json"))


def _extract_case_identifier(dataset: str, filename: str) -> str | None:
    if BENCHMARK == "VIVABENCH":
        pattern = re.compile(rf"{re.escape(dataset)}_(.+?)_conversation_\d+\.json")
    else:
        pattern = re.compile(rf"{re.escape(dataset)}_(\d+)_conversation_\d+\.json")

    match = pattern.search(filename)
    if not match:
        return None
    return match.group(1)


def _build_evaluation_input(eval_dataset, case_identifier: str, path: str):
    if BENCHMARK == "VIVABENCH":
        case_data = eval_dataset[case_identifier]
        return VivaBenchEvaluationInput(case_identifier, case_data, path)

    hadm_id = int(case_identifier)
    patient_data = eval_dataset[hadm_id]
    return EvaluationInput(hadm_id, patient_data, path)


def _get_input_identifier(input_obj) -> str:
    """Use one stable identifier for both MIMIC hadm_id and VivaBench case_id."""
    return str(getattr(input_obj, "hadm_id", getattr(input_obj, "case_id", "")))


def _load_completed_case_ids(result_path: str) -> set[str]:
    """Read already-written diagnosis results so evaluation can resume instead of restarting."""
    completed: set[str] = set()
    if not os.path.exists(result_path):
        return completed

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            case_id = row.get("hadm_id")
            if case_id is not None:
                completed.add(str(case_id))

    return completed


def _filter_missing_inputs(inputs: list, result_path: str) -> tuple[list, int]:
    """Keep only inputs that are not already present in the output jsonl."""
    completed_ids = _load_completed_case_ids(result_path)
    if not completed_ids:
        return inputs, 0

    filtered = [input_obj for input_obj in inputs if _get_input_identifier(input_obj) not in completed_ids]
    return filtered, len(completed_ids)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model diagnoses.")
    parser.add_argument("--dataset", default="all", help="Dataset name, e.g., appendicitis")
    parser.add_argument("--output", default=_default_output_root(), help="Directory to save logs")
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="Optional: Specify a run ID to evaluate a specific simulation run.",
    )
    parser.add_argument(
        "--conv_root",
        default=_default_conv_root(),
        help="Root directory for conversation files",
    )
    parser.add_argument("--mode", choices=["online", "batch"], default="online", help="Diagnosis evaluation mode.")
    parser.add_argument(
        "--concurrency-limit",
        type=int,
        default=10,
        help="Number of concurrent API requests to send at a time.",
    )
    parser.add_argument(
        "--batch-poll-interval",
        type=int,
        default=30,
        help="Polling interval in seconds for batch jobs.",
    )

    return parser.parse_args()


async def main_async():
    args = parse_args()
    dataset_list = _resolve_dataset_list(args.dataset)
    base_conv_root = args.conv_root
    base_log_dir = args.output
    run_id = args.run_id
    mode = args.mode
    concurrency_limit = args.concurrency_limit
    batch_poll_interval = args.batch_poll_interval

    if run_id is not None:
        base_log_dir = os.path.join(base_log_dir, f"run_{run_id}")
        base_conv_root = os.path.join(base_conv_root, f"run_{run_id}")

    for dataset in dataset_list:
        category = _resolve_category(dataset)
        current_log_dir = os.path.join(base_log_dir, dataset)
        current_conv_dir = os.path.join(base_conv_root, dataset)

        log_message = (
            f"INFO: Evaluating dataset: {dataset}."
            f" Conversation source: {current_conv_dir}"
            f" Logs will be saved in: {current_log_dir}"
        )
        if run_id is not None:
            log_message = f"INFO: Evaluating Run ID: {run_id}, " + log_message.split("INFO: ")[1]

        logger.info(log_message)
        os.makedirs(current_log_dir, exist_ok=True)

        output_path = os.path.join(current_log_dir, f"{dataset}_diagnosis_results.jsonl")
        json_files = _find_conversation_files(current_conv_dir, dataset)
        eval_dataset = _load_eval_dataset(dataset)

        if mode == "batch":
            logging.info(
                "Starting batch diagnosis evaluation for %s with poll interval %ss...",
                dataset,
                batch_poll_interval,
            )
        else:
            logging.info(
                "Starting asynchronous diagnosis evaluation for %s with concurrency limit %s...",
                dataset,
                concurrency_limit,
            )

        async_client = AsyncOpenAI()
        evaluator = DiagnosisEvaluator(async_client, EVALUATOR_MODEL, category, output_path)

        all_inputs = []
        logging.info("Preparing %s evaluation inputs...", len(json_files))
        for path in tqdm(json_files, desc="Preparing Inputs"):
            case_identifier = _extract_case_identifier(dataset, os.path.basename(path))
            if case_identifier is None:
                continue

            try:
                all_inputs.append(_build_evaluation_input(eval_dataset, case_identifier, path))
            except KeyError:
                logging.warning("Case identifier %s not found in the dataset. Skipping.", case_identifier)
            except Exception as e:
                logging.error("Error preparing data for case %s: %s", case_identifier, e)

        total_inputs = len(all_inputs)
        all_inputs, completed_count = _filter_missing_inputs(all_inputs, output_path)
        logging.info(
            "Diagnosis evaluation status for %s: total=%s, completed=%s, remaining=%s",
            dataset,
            total_inputs,
            completed_count,
            len(all_inputs),
        )

        if not all_inputs:
            logging.info("No missing cases found for %s. Skipping diagnosis evaluation.", dataset)
            await async_client.close()
            continue

        if mode == "batch":
            batch_dir = os.path.join(current_log_dir, "batch")
            os.makedirs(batch_dir, exist_ok=True)
            request_path = os.path.join(batch_dir, f"{dataset}_diagnosis_batch_requests.jsonl")
            batch_id = await evaluator.evaluate_batch(
                all_inputs,
                request_path=request_path,
                poll_interval=batch_poll_interval,
                metadata={
                    "dataset": dataset,
                    "mode": mode,
                    "run_id": str(run_id) if run_id is not None else "none",
                },
            )
            if batch_id:
                logging.info("Batch evaluation finished for %s. Batch ID: %s", dataset, batch_id)
        else:
            logging.info("Starting concurrent evaluation of %s inputs...", len(all_inputs))
            for i in tqdm(
                range(0, len(all_inputs), concurrency_limit),
                desc=f"Evaluating in Chunks of {concurrency_limit}",
            ):
                chunk = all_inputs[i : i + concurrency_limit]
                await evaluator.evaluate(chunk)

        await async_client.close()
        evaluator.summary()


if __name__ == "__main__":
    asyncio.run(main_async())
