import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def parse_clinicalcase(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Normalize the serialized VivaBench clinicalcase payload into a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"Unsupported clinicalcase payload type: {type(raw).__name__}")


@dataclass
class VivaBenchCase:
    """Lightweight case object used by the simulation and evaluation layers."""

    case_id: str
    source: str
    specialty_group: str
    vignette: str
    clinicalcase_raw: dict[str, Any]

    @property
    def chief_complaint(self) -> str:
        return self.history.get("chief_complaint", "")

    @property
    def history_freetext(self) -> str:
        return self.clinicalcase_raw.get("history_freetext", "")

    @property
    def demographics(self) -> dict[str, Any]:
        return self.clinicalcase_raw.get("demographics", {})

    @property
    def medications(self) -> list[Any]:
        return self.clinicalcase_raw.get("medications", [])

    @property
    def allergies(self) -> list[Any]:
        return self.clinicalcase_raw.get("allergies", [])

    @property
    def history(self) -> dict[str, Any]:
        return self.clinicalcase_raw.get("history", {})

    @property
    def physical(self) -> dict[str, Any]:
        return self.clinicalcase_raw.get("physical", {})

    @property
    def investigations(self) -> dict[str, Any]:
        return self.clinicalcase_raw.get("investigations", {})

    @property
    def imaging(self) -> dict[str, Any]:
        return self.clinicalcase_raw.get("imaging", {})

    @property
    def diagnoses(self) -> list[dict[str, Any]]:
        return self.clinicalcase_raw.get("diagnosis", [])

    @property
    def differential_diagnoses(self) -> list[dict[str, Any]]:
        return self.clinicalcase_raw.get("differentials", [])


class VivaBenchDataset:
    """Simple in-memory dataset wrapper for the benchmark_pubmed split."""

    def __init__(self, cases: list[VivaBenchCase]):
        self.cases = cases
        self.case_ids = [case.case_id for case in cases]
        self._case_by_id = {case.case_id: case for case in cases}

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[VivaBenchCase]:
        return iter(self.cases)

    def __getitem__(self, case_id: str) -> VivaBenchCase:
        return self._case_by_id[case_id]

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "VivaBenchDataset":
        """Load the exported JSONL file and parse each row into a VivaBenchCase."""
        path = Path(path)
        cases: list[VivaBenchCase] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                cases.append(
                    VivaBenchCase(
                        case_id=row["uid"],
                        source=row.get("source", ""),
                        specialty_group=row.get("specialty_group", ""),
                        vignette=row.get("vignette", ""),
                        clinicalcase_raw=parse_clinicalcase(row["clinicalcase"]),
                    )
                )

        return cls(cases)
