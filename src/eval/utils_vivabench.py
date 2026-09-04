from typing import Any


def get_vivabench_ground_truth_diagnosis(case_data: Any) -> list[str]:
    """Extract the gold diagnosis names from a VivaBenchCase."""
    return [entry.get("name", "").strip() for entry in case_data.diagnoses if entry.get("name")]


def get_vivabench_primary_diagnosis(case_data: Any) -> str | None:
    diagnoses = get_vivabench_ground_truth_diagnosis(case_data)
    return diagnoses[0] if diagnoses else None


def get_vivabench_relevant_keys(case_data: Any) -> list[str]:
    """Collect evidence keys attached to the gold diagnoses for later tool/evidence analysis."""
    keys: list[str] = []
    for entry in case_data.diagnoses:
        keys.extend(entry.get("relevant_keys", []))
    return keys


def get_vivabench_ground_truth_package(case_data: Any) -> dict[str, list[str]]:
    """Return a structured gold package for VivaBench judge prompts."""
    gold_differentials = [
        entry.get("name", "").strip()
        for entry in case_data.differential_diagnoses
        if entry.get("name")
    ]
    return {
        "gold_diagnoses": get_vivabench_ground_truth_diagnosis(case_data),
        "gold_differentials": gold_differentials,
        # "relevant_keys": get_vivabench_relevant_keys(case_data),
    }
