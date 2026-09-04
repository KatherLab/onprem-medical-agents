from dataclasses import dataclass
from typing import Any

from .utils import get_assistant_diagnosis
from .utils_vivabench import get_vivabench_ground_truth_package



@dataclass
class VivaBenchEvaluationInput:
    case_id: str
    case_data: Any
    conversation_path: str

    def ground_truth_diagnosis(self, match_criterion: str | None = None) -> dict[str, list[str]]:
        return get_vivabench_ground_truth_package(self.case_data)

    def assistant_diagnosis(self) -> dict:
        return get_assistant_diagnosis(self.conversation_path)
