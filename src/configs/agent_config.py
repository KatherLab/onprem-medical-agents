import os
import logging
from dotenv import load_dotenv
from typing import Literal, Dict, List

load_dotenv()


def _get_env_with_fallback(primary_key: str, fallback_key: str) -> str | None:
    return os.getenv(primary_key) or os.getenv(fallback_key)

# logging level
# LOG_LEVEL = logging.ERROR
LOG_LEVEL = logging.INFO

# -------------Simulate--------------
# qdrant setup
QDRANT_ICD_PROCEDURE_COLLECTION_NAME: str = "mimic_iv_icd_codes_procedures"
QDRANT_ICD_PROCEDURE_EMBEDDING_MODEL: str = "jinaai/jina-embeddings-v3"
EMBEDDING_MODEL_DIR = "/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/checkpoints/jina"
QDRANT_URL: str = "http://localhost:6333"

BENCHMARK: str = "AIDOC"  #"VIVABENCH"#"CDM"
FORMATTED_OUTPUT_ROOT: str = "/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/formatted_outputs"

VIVABENCH_DATA_PATH: str = "/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/vivabench/benchmark_pubmed_test.jsonl"
VIVABENCH_BLOOD_MATCHER_MODEL: str = "GLM-4.5-Air-FP8" #"Qwen3.5-397B-A17B-FP8"
VIVABENCH_BLOOD_MATCHER_TEMPERATURE: float = 0.0

DATASET_NAMES: list[str] = [
    "appendicitis",
    "cholecystitis", 
    "diverticulitis", 
    "pancreatitis", 
    "pneumonia", 
    "pulmonary_embolism", 
    "uti",
]
MAX_SAMPLES: int | None = None  # if set to None, we use all samples
MAX_EVAL_PARALLEL_SIZE: int | None = (
    1  # maximum number of parallel evaluations (not in interactive mode)
)

# med assistant settings
MEDICAL_ASSISTANT_MODEL: str ="litellm/openai/Qwen3.5-397B-A17B-FP8" #"litellm/openai/GLM-4.5-Air-FP8" # "litellm/openai/llama-3.3-70b-instruct" #"google/gemini-3.1-flash-lite-preview" #"litellm/openai/gemma-4-31B-it" #"litellm/openai/GPT-OSS-120B" #"litellm/openai/GLM-5-FP8" #"litellm/openai/Kimi-K2.5" #"litellm/openai/GPT-OSS-120B" #"litellm/openai/GLM-4.7-FP8" #"gpt-5.2-2025-12-11" #"litellm/openai/gpt-4o-2024-08-06" #"gpt-4o" #"litellm/openai/medgemma-27b-it-q6" #"gpt-4o" #"gpt-4.1" #"litellm/openai/GLM-4.5V" #"litellm/openai/llama-3.1-8b-instruct-q8" #"litellm/openai/llama-3.3-70b-instruct-awq"
 #"Kimi-K2-Instruct" #"Llama-3.3-70B-Instruct" #"medgemma-4b-it-q8" #"meta-llama-3.1-8b-instruct-q8" #"Qwen3-235B-A22B-FP8" #"llama-3.3-70b-instruct-q4km" #"Qwen2.5-VL-72B-Instruct" #"Llama-4-Scout-17B-16E-Instruct" #"deepseek-r1-distill-qwen-32b-q8"
MEDICAL_ASSISTANT_TEMPERATURE: float = 0.6
MEDICAL_ASSISTANT_BASE_URL: str | None = _get_env_with_fallback(
    "MEDICAL_ASSISTANT_BASE_URL", "OPENAI_BASE_URL"
)
MEDICAL_ASSISTANT_API_KEY: str | None = _get_env_with_fallback(
    "MEDICAL_ASSISTANT_API_KEY", "OPENAI_API_KEY"
)

# patient assistant settings
PATIENT_ASSISTANT_MODEL: str = "litellm/openai/Qwen3.5-397B-A17B-FP8"
PATIENT_ASSISTANT_TEMPERATURE: float = 0.01
PATIENT_ASSISTANT_BASE_URL: str | None = _get_env_with_fallback(
    "PATIENT_ASSISTANT_BASE_URL", "OPENAI_BASE_URL"
)
PATIENT_ASSISTANT_API_KEY: str | None = _get_env_with_fallback(
    "PATIENT_ASSISTANT_API_KEY", "OPENAI_API_KEY"
)

CRITIC_MODEL: str = "litellm/openai/GPT-OSS-120B" #"litellm/openai/medgemma-27b-it-fp8-dynamic" #"litellm/openai/medgemma-27b-it-q6" #"gpt-4o-mini"
EXECUTOR_MODEL: str = "litellm/openai/GLM-4.5-Air-FP8"

# reasoning model
REASONING_MODEL: str = (
    "litellm/openai/GPT-OSS-120B" #"Qwen3-235B-A22B-Thinking-2507-FP8" #"o3" #"deepseek-r1-distill-qwen-32b-q8"
)


AIDOC_HADM_IDS: Dict[str, List[int]] = {
    "appendicitis": [
        92000001,
    ],
    "cholecystitis": [
        21210624,
    ],
    "diverticulitis": [
        29381122,
    ],
    "pancreatitis": [
        28079494,
    ],
    "pneumonia": [
        28519937,
    ],
    "pulmonary_embolism": [
        26406913,
    ],
    "uti": [
        26514952,
    ],
}


CDM_HADM_IDS: Dict[str, List[int]] = {
    "appendicitis": [
        20890008,
    ],
    "cholecystitis": [
        29897948,
    ],
    "diverticulitis": [
        24911566,
    ],
    "pancreatitis": [
        29103261,
    ],
}

#--------------------Evaluate----------------
# patient assistant settings
EVALUATOR_MODEL: str = "google/gemini-3.1-flash-lite-preview" #"openai/gpt-4o-2024-08-06" #"gpt-4o-2024-08-06" 
