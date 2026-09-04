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
    # Attempt to parse the value.
    value = None

    observation_id = str(uuid.uuid4())  # Generate a unique ID.

    if result is None:
        note = f"No lab result available for {lab_request.lab_value.value}. Do not request this test again."
        # note = f"No lab result available for {lab_request.lab_value.value}."
        return {
            "id": observation_id,
            "status": "missing",
            "lab_value": lab_request.lab_value,
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "message": "No lab test result available. ",
            "note": note,
        }


    if valid(result.get("valuenum")):
        try:
            value = float(result["valuenum"])
        # except ValueError:
        except (ValueError, TypeError):
            value = None
    elif valid(result.get("value")):
        try:
            value = float(result["value"])
        # except ValueError:
        except (ValueError, TypeError):
            # value = result["value"]
            value = result.get("value", None)  # Or None.
    elif valid(result.get("flag")):
        value = result["flag"]
    else:
        value = None  # All parsing branches failed.

    if value is None or value == "None":
        note = f"{result['label']} result unavailable or unmeasurable."
        return {
            "id": observation_id,
            "status": "invalid",
            "lab_value": result["label"],
            "patient_id": patient_id,
            "timestamp": result["charttime"].isoformat(),
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

    # Unit.
    unit = result.get("valueuom_x")
    if not unit or unit == "None":
        unit = "unit not reported"

    note = f"{result['label']}: {value} {unit} (Reference: {ref_range})"
    # print('note:', note)

    # Final result.
    return {
        "id": observation_id,
        "status": "final",
        "lab_value": result["label"],
        "patient_id": patient_id,
        "timestamp": result["charttime"].isoformat(),
        "value": value,
        "unit": unit,
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
            "timestamp": datetime.now().isoformat(),
            "message": "No urine test result available.",
            "note": f"No urine test for {urine_request.urine_value.value} is available. Do not request this test again.",
        }

    # Parse the value.
    value = None
    if valid(result.get("valuenum")):
        try:
            value = float(result["valuenum"])
        except ValueError:
            value = None
    elif valid(result.get("value")):
        try:
            value = float(result["value"])
        except ValueError:
            value = result["value"]
    elif valid(result.get("flag")):
        value = result["flag"]

    if value is None or value == "None":
        fallback_val = (
            result.get("valuenum") or
            result.get("value") or
            result.get("flag") or
            "N/A"
        )
        return {
            "id": observation_id,
            "status": "invalid",
            "lab_value": result["label"],
            "patient_id": patient_id,
            "timestamp": result["charttime"].isoformat(),
            "message": "Urine test result data is invalid.",
            # "note": f"{result['label']}: {fallback_val} (data missing or invalid)",
            "note": f"{result['label']}: result unavailable or unmeasurable.",
        }

    # Reference range.
    ref_low = result.get("ref_range_lower")
    ref_high = result.get("ref_range_upper")
    if pd.isna(ref_low) or pd.isna(ref_high):
        ref_range = "reference range not available"
    else:
        ref_range = f"{ref_low} - {ref_high}"

    # Unit.
    unit = result.get("valueuom_x")
    if not unit or unit == "None":
        unit = "unit not reported"

    note = f"{result['label']}: {value} {unit} (Reference: {ref_range})"

    return {
        "id": observation_id,
        "status": "final",
        "lab_value": result["label"],
        "patient_id": patient_id,
        "timestamp": result["charttime"].isoformat(),
        "value": value,
        "unit": unit,
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
            f"{result['frequency']} x {result['period']}{result['period_unit'].name}, "
            f"{result['route'].value}"
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


def generate_micro_test_observation_resource(
    microbiology_request: MicrobiologyRequest,
    result: Union[pd.Series, None],
    patient_id: str,
) -> Dict[str, Any]:
    observation_id = str(uuid.uuid4())

    if result is None: 
        note = f"{microbiology_request.microbiology_value.value}: No microbiology result available. Do not request this test again."
        return {
            "id": observation_id,
            "status": "missing",
            "microbiology_test": microbiology_request.microbiology_value.value,
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "message": "No microbiology test results available.",
            "note": note,
        }

    result_str = result["grouped_microbio_str"].unique()[0] if "grouped_microbio_str" in result else "Result not available."
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



def generate_micro_org_observation_resource(
    microbiology_request: MicrobiologyRequest,
    result: pd.Series,
    patient_id: str,
) -> Dict[str, Any]:
    observation_id = str(uuid.uuid4())

    org_id = (
        str(result["org_itemid"].unique()[0])
        if "org_itemid" in result and pd.notna(result["org_itemid"].unique()[0])
        else "Unknown Organism"
    )
    org_name = (
        str(result["org_name"].unique()[0])
        if "org_name" in result and pd.notna(result["org_name"].unique()[0])
        else "Unknown Organism"
    )

    return {
        "id": observation_id,
        "status": "final",
        "microbiology_test": microbiology_request.microbiology_value.value,
        "patient_id": patient_id,
        "timestamp": datetime.now().isoformat(),
        "organism_id": org_id,
        "organism_name": org_name.strip(),
        "note": f"Organism identified: {org_name.strip()} (from {microbiology_request.microbiology_value.value})"
    }


def generate_micro_susc_observation_resource(
    microbiology_request: MicrobiologyRequest,
    antibiotic_result: dict,
    patient_id: str,
) -> Dict[str, Any]:
    observation_id = str(uuid.uuid4())

    ab_id = str(antibiotic_result.get("ab_itemid", "unknown"))
    ab_name = antibiotic_result.get("ab_name", "Unknown Antibiotic")
    interpretation = antibiotic_result.get("interpretation", "unknown")
    dilution_value = antibiotic_result.get("dilution_value", None)

    note = f"{ab_name.strip()} - {interpretation.strip()}"
    if dilution_value is not None:
        note += f" (MIC: {dilution_value})"

    return {
        "id": observation_id,
        "status": "final",
        "microbiology_test": microbiology_request.microbiology_value.value,
        "patient_id": patient_id,
        "timestamp": datetime.now().isoformat(),
        "antibiotic_id": ab_id,
        "antibiotic_name": ab_name.strip(),
        "interpretation": interpretation.strip(),
        "dilution_value": dilution_value if dilution_value is not None else None,
        "note": note + f" (from {microbiology_request.microbiology_value.value})"
    }



def generate_microbiology_observations(
    microbiology_request: MicrobiologyRequest,
    result: Union[pd.DataFrame, None],
    patient_id: str,
) -> List[Dict[str, Any]]:
    """
    Generate microbiology data (cultures and antibiotic susceptibility) in a non-FHIR format.

    Args:
        microbiology_request (MicrobiologyRequest): Microbiology test request object.
        result (pd.DataFrame | None): Microbiology test DataFrame, which may be empty.
        patient_id (str): Patient identifier.

    Returns:
        List[Dict[str, Any]]: Observations for microbiology tests, organisms, and antibiotic susceptibility.
    """

    # Generate the parent microbiology test observation.
    micro_test_observation = generate_micro_test_observation_resource(
        microbiology_request, result, patient_id
    )

    observations = [micro_test_observation]
    # print('observations:', observations)

    # Return immediately when no microbiology result is available.
    if result is None or result.empty:
        return observations

    # Require an org_name column with at least one non-null value before processing organisms.
    if "org_name" not in result.columns or result["org_name"].isnull().all():
        return observations  # No organism was detected.

    # Iterate over each distinct organism.
    unique_org_names = result["org_name"].dropna().unique()
    for org_name in unique_org_names:
        org_df = result[result["org_name"] == org_name]

        # Generate the organism observation.
        micro_org_observation = generate_micro_org_observation_resource(
            microbiology_request,
            org_df,
            patient_id
        )
        observations.append(micro_org_observation)

        # Generate antibiotic susceptibility observations.
        for _, antibiotic_result in org_df.iterrows():
            if pd.notnull(antibiotic_result.get("ab_name")):
                micro_susc_observation = generate_micro_susc_observation_resource(
                    microbiology_request,
                    antibiotic_result,
                    patient_id
                )
                observations.append(micro_susc_observation)

    return observations


################################################################################


def generate_radiology_report_resource(
    radiology_request: RadiologyRequest,
    result: Union[pd.Series, dict, str, None],
    patient_id: str,
) -> Dict[str, Any]:
    """
    Generates a simplified radiology report (non-FHIR format) with a formatted text note.

    Args:
        radiology_request (RadiologyRequest): Radiology test request object.
        result (Union[pd.Series, dict, str, None]): The extracted report data or None.
        patient_id (str): Patient ID.

    Returns:
        Dict[str, Any]: A dictionary containing radiology report metadata and a human-readable 'note'.
    """

    report_id = str(uuid.uuid4())
    study_time = datetime.now().isoformat()

    if result is None:
        # Missing report case
        report_text = "Radiology Report:\n    Examination could not be performed. Do not request this test again."
        status = "unknown"
        conclusion_text = "Radiology report not available."
    else:
        # Parse available report
        if isinstance(result, dict):
            report_text = result.get("extracted_rad_events", "Report data is unavailable.")
            study_time = result.get("charttime", datetime.now()).isoformat()
        elif isinstance(result, pd.Series):
            report_text = result.get("extracted_rad_events", "Report data is unavailable.")
            study_time = result.get("charttime", datetime.now()).isoformat()
        elif isinstance(result, str):
            report_text = result
        else:
            report_text = "Report format not recognized."

        status = "final"
        conclusion_text = None

    # Construct LLM-friendly note
    llm_input = (
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
        "note": llm_input  # ✅ Add this for LLM usage
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
        "\n\nCall the 'request_procedure' tool with the exact name if you'd like to perform one of these."
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
        "patient_id": patient_id,
        "timestamp": datetime.now().isoformat(),
        "note": note,
    }


