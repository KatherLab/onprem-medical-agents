from dataclasses import dataclass
from typing import Any

from .utils import get_assistant_diagnosis, get_ground_truth_diagnosis



@dataclass
class EvaluationInput:
    hadm_id: int
    patient_data: Any
    conversation_path: str

    def ground_truth_diagnosis(self, match_criterion: str) -> list[str]:
        return get_ground_truth_diagnosis(self.patient_data, match_criterion)

    def assistant_diagnosis(self) -> dict:
        return get_assistant_diagnosis(self.conversation_path)
