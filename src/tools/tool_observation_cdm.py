import uuid
from datetime import datetime
from typing import List, Optional, Union, Dict, Any

import pandas as pd

from .tool_request import *



def valid(item):
    # check if item is not NaN and not "___"
    return pd.notna(item) and item != "___"


################################################################################


def generate_lab_observation_resource(
    lab_request: LabRequest,
    result: Union[pd.Series, None],
    patient_id: str,
) -> Dict[str, Any]:
    """
    Generate laboratory test data in a non-FHIR format.

    Args:
        lab_request (LabRequest): Blood test request object.
        result (pd.Series | None): Test result data, or `None` when unavailable.
        patient_id (str): Patient identifier.

    Returns:
        Dict[str, Any]: Laboratory test data, with a `message` when data is missing.
    """

    observation_id = str(uuid.uuid4())  # Generate a unique ID.
    if result is None:
        note = f"No lab result available for {lab_request.lab_value.value}. Do not request this test again."
        return {
            "id": observation_id,
            "status": "missing",
            "lab_value": lab_request.lab_value,
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "message": "No lab test result available. ",
            "note": note,
        }
    
    value = None
    # Attempt to parse the value.
    if valid(result.get("value")):
        try:
            value = result["value"]
        except ValueError:
            value = None

    if value is None or value == "None":
        note = f"{result['label']} result unavailable or unmeasurable"
        return {
            "id": observation_id,
            "status": "invalid",
            "lab_value": result["label"],
            "patient_id": patient_id,
            "message": "Lab result data is invalid.",
            "note": note,
        }

    # Reference range.
    ref_low = result.get("ref_range_lower")
    ref_high = result.get("ref_range_upper")
    if pd.isna(ref_low) or pd.isna(ref_high):
        ref_range = "reference range not available"
    else:
        ref_range = f"{ref_low} - {ref_high}"

    note = f"{result['label']}: {value} (Reference: {ref_range})"
    # print('note:', note)

    # Final result.
    return {
        "id": observation_id,
        "status": "final",
        "lab_value": result["label"],
        "patient_id": patient_id,
        "value": value,
        "reference_range": f"{ref_range}",
        "note": note,
    }



################################################################################



def generate_urine_observation_resource(
    urine_request: UrineRequest,
    result: Union[pd.Series, None],
    patient_id: str,
) -> Dict[str, Any]:
    """
    Generate urine test data in a non-FHIR format with a natural-language note field.

    Args:
        urine_request (UrineRequest): Urine test request object.
        result (pd.Series | None): Test result data, or `None` when unavailable.
        patient_id (str): Patient identifier.

    Returns:
        Dict[str, Any]: Observation data including a note field.
    """
    observation_id = str(uuid.uuid4())

    if result is None:
        return {
            "id": observation_id,
            "status": "missing",
            "lab_value": urine_request.urine_value,
            "patient_id": patient_id,
            "message": "No urine test result available.",
            "note": f"No urine test for {urine_request.urine_value.value} is available. Do not request this test again.",
        }

    # Parse the value.
    if valid(result.get("value")):
        try:
            value = result["value"]
        except ValueError:
            value = None

    if value is None or value == "None":
        note = f"{result['label']} result unavailable or unmeasurable"
        return {
            "id": observation_id,
            "status": "invalid",
            "lab_value": result["label"],
            "patient_id": patient_id,
            "message": "Urine test result data is invalid.",
            "note": f"{result['label']}: result unavailable or unmeasurable",
        }

    # Reference range.
    ref_low = result.get("ref_range_lower")
    ref_high = result.get("ref_range_upper")
    if pd.isna(ref_low) or pd.isna(ref_high):
        ref_range = "reference range not available"
    else:
        ref_range = f"{ref_low} - {ref_high}"

    note = f"{result['label']}: {value}(Reference: {ref_range})"

    return {
        "id": observation_id,
        "status": "final",
        "lab_value": result["label"],
        "patient_id": patient_id,
        "value": value,
        "reference_range": ref_range,
        "note": note,
    }



################################################################################


def generate_medication_observation_resource(
    medication_request: MedicationRequest,
    result: Optional[pd.Series],
    patient_id: str,
) -> Dict[str, Any]:
    """
    Generates a non-FHIR medication observation dictionary for LLM consumption.

    Args:
        medication_request (MedicationRequest): The medication request object.
        result (pd.Series | None): Medication result data from the dataset.
        patient_id (str): The ID of the patient.

    Returns:
        Dict[str, Any]: A dictionary representing the medication observation.
    """

    observation_id = str(uuid.uuid4())

    if result is None:
        return {
            "id": observation_id,
            "status": "missing",
            "medication_name": medication_request.drug_name,
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "message": "No medication record available.",
            "note": f"{medication_request.drug_name}: No medication record found.",
        }

    # Extract data and construct a natural-language note.
    try:
        note = (
            f"{result['drug_name']}: {result['dosage_text']} "
            f"{result['dosage_value']} {result['dosage_unit']}, "
            f"{result['frequency']} x {result['period']}{result['period_unit']}, "
            f"{result['route']}"
        )
    except Exception:
        note = "Medication result formatting failed."

    try:
        dosage_value = float(result["dosage_value"])
    except (ValueError, TypeError):
        dosage_value = None

    dosage_unit = str(result["dosage_unit"]).strip() if result.get("dosage_unit") else None
    issued_time = result["issued"].isoformat() if isinstance(result.get("issued"), datetime) else datetime.now().isoformat()

    return {
        "id": observation_id,
        "status": "final",
        "medication_name": result["drug_name"],
        "patient_id": patient_id,
        "timestamp": issued_time,
        "dosage_value": dosage_value,
        "dosage_unit": dosage_unit,
        "frequency": result.get("frequency"),
        "period": result.get("period"),
        "period_unit": result.get("period_unit"),
        "route": result.get("route"),
        "note": note,
    }


################################################################################


def generate_pe_observation_resource(
    pe_request: PhysicalExamRequest,
    result: Optional[str],  # or None
    patient_id: str,
) -> Dict[str, Any]:
    """
    Generates a simplified physical examination observation (non-FHIR).

    Args:
        pe_request (PhysicalExamRequest): The physical exam request object.
        result (Optional[str]): The physical examination result (or None).
        patient_id (str): The patient ID.

    Returns:
        Dict[str, Any]: A structured dict with a clean 'note' for downstream use.
    """

    observation_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    if result is None:
        note = "Physical Examination:\n   Physical examination not available. Do not request this test again."
        return {
            "id": observation_id,
            "status": "missing",
            "patient_id": patient_id,
            "timestamp": timestamp,
            "message": "No physical examination data available.",
            "note": note,
        }

    cleaned_result = result.strip()
    note = f"Physical Examination:\n   {cleaned_result}"

    return {
        "id": observation_id,
        "status": "final",
        "patient_id": patient_id,
        "timestamp": timestamp,
        "exam_result": cleaned_result,
        "note": note,
    }


################################################################################


def generate_microbiology_observations(
    microbiology_request: MicrobiologyRequest,
    result: Union[pd.Series, None],
    patient_id: str,
) -> Dict[str, Any]:
    observation_id = str(uuid.uuid4())

    if result is None: 
        note = f"{microbiology_request.microbiology_value.value}: No microbiology result available."
        return {
            "id": observation_id,
            "status": "missing",
            "microbiology_test": microbiology_request.microbiology_value.value,
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "message": "No microbiology test results available.",
            "note": note,
        }

    result_str = result.get("grouped_microbio_str", "Result not available")

    # print('result_str:', result_str)
    note = f"{microbiology_request.microbiology_value.value}:\n  {result_str.strip()}"
    # print('note_test:', note)

    return {
        "id": observation_id,
        "status": "final",
        "microbiology_test": microbiology_request.microbiology_value.value,
        "patient_id": patient_id,
        "timestamp": datetime.now().isoformat(),
        "test_result": result_str.strip(),
        "note": note,
    }


################################################################################


def generate_radiology_report_resource(
    radiology_request: RadiologyRequest,
    result: Union[pd.Series, dict, str, None],
    patient_id: str,
) -> Dict[str, Any]:
    """
    Generates a simplified radiology report with formatted text.
    """

    report_id = str(uuid.uuid4())
    study_time = datetime.now().isoformat()

    if result is None:
        report_text = "Radiology Report:\n    Examination could not be performed. Do not request this test again."
        status = "unknown"
        conclusion_text = "Radiology report not available."
    else:
        # Accept Series or dict
        if isinstance(result, (dict, pd.Series)):
            report_text = result.get("extracted_rad_events", "Report data is unavailable.")
        elif isinstance(result, str):
            report_text = result
        else:
            report_text = "Report format not recognized."

        status = "final"
        conclusion_text = None

    note = (
        f"Radiology Report ({radiology_request.modality.value}, {radiology_request.region.value}):\n\n"
        f"{report_text}"
    )

    return {
        "id": report_id,
        "status": status,
        "modality": radiology_request.modality.value,
        "region": radiology_request.region.value,
        "patient_id": patient_id,
        "timestamp": study_time,
        "report_text": report_text,
        "conclusion": conclusion_text,
        "note": note
    }



################################################################################



def generate_procedure_search_resource(
    procedure_request: ProcedureSearch,
    result: Union[List[dict], None],
    patient_id: str,
) -> Dict[str, str]:
    """
    Generates a simplified procedure search result.

    Args:
        procedure_request: Procedure search request object.
        result (List[dict] | None): Results returned from vector search.
        patient_id (str): The ID of the patient.

    Returns:
        Dict[str, str]: A dictionary with procedure info and notes.
    """
    procedure_id = str(uuid.uuid4())

    if not result:
        note = f"No matching procedures found for query '{procedure_request.procedure}'."
        return {
            "id": procedure_id,
            "status": "missing",
            "procedure_query": procedure_request.procedure,
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "note": note,
        }

    # formatted_options = [
    #     f"- {opt.payload['long_title']}" for opt in result.points
    # ]
    formatted_options = []
    for p in result.points:
        title = p.payload.get("long_title", "Unknown Procedure")
        formatted_options.append(f"- {title}")

    note = (
        f"Top procedure options for query '{procedure_request.procedure}':\n" +
        "\n".join(formatted_options) +
        "\n\nCall the ProcedureRequest tool with the exact name if you'd like to perform one of these."
    )

    return {
        "id": procedure_id,
        "status": "options",
        "procedure_query": procedure_request.procedure,
        "patient_id": patient_id,
        "timestamp": datetime.now().isoformat(),
        "note": note,
    }



# To use the code below, update fetch_procedure_request_results(); otherwise it returns a string and causes "'str' object has no attribute 'points'".
def generate_procedure_resource(
    procedure_request: ProcedureRequest,
    result: Union[List[dict], None],
    patient_id: str,
) -> Dict[str, str]:
    """
    Generates a simplified procedure confirmation result.

    Args:
        procedure_request: The requested procedure.
        result (List[dict] | None): Procedure match results.
        patient_id (str): The ID of the patient.

    Returns:
        Dict[str, str]: Simplified dictionary describing the procedure.
    """
    procedure_id = str(uuid.uuid4())

    display = procedure_request.procedure

    if not result:
        # code = "unknown"
        note = f"Procedure:\n    Procedure could not be documented in the system."
    else:
        # top_match = result.points[0].payload
        # code = top_match.get("icd_code", "_123")
        # display = top_match.get("long_title", procedure_request.procedure)
        # note = f"Procedure '{display}' (ICD: {code}) has been requested and documented for this patient."

        # top_match = result[0]
        # display = top_match.get("long_title", procedure_request.procedure)
        note = f"Requested procedure for '{display}' on the system.\n\n"

    return {
        "id": procedure_id,
        "status": "completed",
        "procedure_name": display,
        # "procedure_code": code,
        "patient_id": patient_id,
        "timestamp": datetime.now().isoformat(),
        "note": note,
    }


