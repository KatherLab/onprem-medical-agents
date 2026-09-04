from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from ..MimicEnums import *

def make_empty_usage_dict() -> dict[str, int]:
    """Return a stable zero-initialized usage payload for per-case accounting."""
    return {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }


def add_usage_inplace(target: dict[str, int], source: Optional[dict[str, int]]) -> None:
    """Merge token accounting into an existing accumulator."""
    if not source:
        return

    for key in make_empty_usage_dict():
        target[key] = target.get(key, 0) + int(source.get(key, 0) or 0)


@dataclass(frozen=False)
class PatientContext:
    """Encapsulates all necessary context for processing patient-specific tool calls."""
    patient_id: str
    patient_hadm_id: str
    patient_data: Any
    patient_info: Optional[str] = None
    benchmark: Optional[str] = None
    case_id: Optional[str] = None
    blood_matcher_usage: dict[str, int] = field(default_factory=make_empty_usage_dict)
    deployment_metadata: Optional[dict[str, Any]] = None


registered_resources = []


def register_class(klass):
    global registered_resources
    registered_resources.append(klass)
    return klass


@register_class
class LabRequest(BaseModel):
    """Request model for a single blood lab test, ensuring only valid lab values are used."""

    lab_value: BloodValue = Field(
        description="The blood value to request, selected from the BloodValue Enum.",
        example=BloodValue._50803, 
    )

class LabRequestList(BaseModel):
    """Request for a list of lab values"""
    lab_values: List[LabRequest] = Field(
        ..., description="The list of lab values to request for the patient."
    )


@register_class
class UrineRequest(BaseModel):
    """Request for a single Urine Value"""

    urine_value: UrineValue = Field(  # TODO: FIXME HARDCODING
        description="The urine value to request, selected from the UrineValue Enum.",
        example=UrineValue._51486,  # TODO: FIXME HARDCODING
    )

class UrineRequestList(BaseModel):
    """Request for a list of urine values"""

    urine_values: List[UrineRequest] = Field(
        ..., description="The list of urine values to request for the patient."
    )


@register_class
class PhysicalExamRequest(BaseModel):
    """Perform a physical examination of a patient."""
    ...


@register_class
class MicrobiologyRequest(BaseModel):
    """Request for a Microbiology Test"""

    microbiology_value: MicroBiologyValue = Field(
        description="The microbiology test to request, selected from the MicroBiologyValue Enum.",
        example=MicroBiologyValue._90144,
    )

class MicrobiologyRequestList(BaseModel):
    """Request for a list of microbiology tests"""

    microbiology_tests: List[MicrobiologyRequest] = Field(
        description="The list of microbiology tests"
    )


@register_class
class RadiologyRequest(BaseModel):
    """Request for a radiology examination. "Venous" Ultrasound refers to a venous ultrasound of the lower extremities (Duplex)."""

    modality: RadiologyModalityValue = Field(
        description="The imaging modality to be used.",
        example=RadiologyModalityValue.CT,
    )
    region: RadiologyRegionValue = Field(
        description="The body region to be imaged.",
        example=RadiologyRegionValue.Abdomen,
    )
    info: Optional[str] = Field(
        default=None,
        description="Any additional clinical information or questions for the radiologist to consider.",
        example="Evaluate for pulmonary embolism.",
    )



# Search candidate options (flexible/free-form).
@register_class
class ProcedureSearch(BaseModel):
    """Search for a procedure and receive a list of options that you can call the `ProcedureRequest` tool with.
    Always search for possible procedures with this tool before using the `ProcedureRequest` tool.
    """

    procedure: str = Field(
        description="Short description of the procedure to perform. This involves therapeutic procedures, like surgeries. For mostly diagnostic procedures like `ERCP` use the `RadiologyRequest` tool.",
        example="laparoscopic cholecystectomy",
    )

# Select an exact option (safe/controlled).
@register_class
class ProcedureRequest(BaseModel):
    """A class representing a procedure request that is requested for a patient."""

    procedure: str = Field(
        description="Exact name of the procedure to perform. Should be called after `ProcedureSearch` tool with one of the options `option` where option is the exact name of the procedure. If the search did not return options you were looking for, try to search again, or skip. This involves therapeutic procedures, like surgeries. For mostly diagnostic procedures like `ERCP` use the `RadiologyRequest` tool.",
        example="Laparoscopic appendectomy",
    )



@register_class
class MedicationRequest(BaseModel):
    """Request for a single medication"""

    # replace by Enums
    drug_name: str = Field(description="The name of the drug", example="Amoxicillin")
    dosage_text: str = Field(  # prod_strength
        description="The dosage, strength or concentration a single medication as text",
        example="10mEq ER Tablet once a day",  # prod strength
    )
    dosage_value: Union[int, float] = Field(  # dose_val_rx
        # dosage_value: Union[int, float, str] = Field(  # dose_val_rx
        description="The prescribed dosage for the patient in one intake",
        example=500,  # improve
    )
    dosage_unit: str = Field(
        description="The unit of the dosage value", example="mg"
    )  # dose_unit_rx
    period: int = Field(description="The period of the dosage", example=1)
    period_unit: PeriodUnit = Field(description="The unit of the period", example="d")
    frequency: int = Field(
        description="The frequency of the dosage per period", example=3
    )  # doses_per_24_hrs
    route: RouteUnit = Field(description="The route of the dosage", example="PO")

class MedicationRequestList(BaseModel):
    """Request for a list of medications"""

    medications: List[MedicationRequest] = Field(
        description="The list of medications"
    )


class Finish(BaseModel):
    """Indicate that the patient case is ready for to be closed in the emergency department, once you have thoroughly completed all necessary diagnostic and therapeutic steps so far."""

    diagnosis: str = Field(
        description="The diagnosis of the patient in short form. Example: `Left sided pneumonia`"
    )

class Think(BaseModel):
    """Used to reflect or plan before proceeding."""
    thought: str = Field(
        description="The internal reflection, reasoning, or plan before taking the next action."
    )


class Plan(BaseModel):
    """Generate a structured sequence of next actions and steps to be taken to complete the patient case."""

    ...


class TalkToPatient(BaseModel):
    """Sends a message to the patient to ask a question or provide information."""
    message: str = Field(..., description="The content of the message to send to the patient.")





