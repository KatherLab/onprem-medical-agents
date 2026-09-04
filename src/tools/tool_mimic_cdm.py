import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..MimicEnums import *
from .tool_request import *
from ..qdrant_collection import Qdrant_Collection



def fetch_lab_results(patient_data: pd.DataFrame, patient_id: str, patient_hadm_id: str, lab_request: LabRequest) -> Dict[str, Any]:
    """
    Retrieves a blood lab test result (no time filtering) for a specific patient and test type.

    Args:
        patient_data (pd.DataFrame): Patient's lab_events DataFrame (without charttime).
        patient_id (str): Patient ID.
        patient_hadm_id (str): Admission ID.
        lab_request (LabRequest): Contains lab test name (lab_value).

    Returns:
        Optional[Dict[str, Any]]: A dictionary with test result info, or None if not found.
    """

    lab_value = lab_request.lab_value  # Ensure lab_value is a valid BloodValue.

    # Filter for the requested blood test and patient admission
    filtered = patient_data[
        (patient_data["label"] == lab_value)
        & (patient_data["fluid"] == "Blood")
    ]    

    if filtered.empty:
        print(
            f"No lab results found for patient_id: {patient_id} and lab_value: {lab_value}.\n"
            "Returning 'None' to indicate a missing lab result."
        )
        return None

    result = filtered.iloc[0]
    return result


def fetch_urine_results(patient_data: pd.DataFrame, patient_id: str, patient_hadm_id: str, urine_request: UrineRequest) -> Dict[str, Any]:
    """
    Retrieves urine lab test results for a specific patient and requested test.

    This function searches the patient's laboratory test records for the specified urine test.
    It returns the earliest recorded result within the first 24 hours of the hospital admission.

    Args:
        patient_data (pd.DataFrame): 
            A DataFrame containing the laboratory test records for the patient.
        patient_id (str): 
            The unique identifier of the patient.
        patient_hadm_id (str): 
            The unique hospital admission ID for the patient's visit.
        urine_value (str): 
            The name of the urine test to retrieve (e.g., "Urine pH", "Specific Gravity").

    Returns:
        Dict[str, Any]: A dictionary containing the earliest available urine test result within
        the first 24 hours of admission.
        If no matching results are found, logs a message and returns `None`.
    """

    urine_value = urine_request.urine_value

    # Filter for the requested urine test and patient admission
    filtered = patient_data[
        (patient_data["label"] == urine_value)
        & (patient_data["fluid"] == "Urine")
    ]

    if filtered.empty:
        print(
            f"No urine test results found for patient_id: {patient_id} and urine_value: {urine_value}.\n"
            "Returning 'None' to indicate a missing urine test result."
        )
        return None

    result = filtered.iloc[0]
    return result


def fetch_pe_results(
    patient_data: pd.DataFrame,
    patient_id: str,
    patient_hadm_id: str | int,
    pe_request: PhysicalExamRequest,
) -> Optional[str]:
    """
    Fetches the physical examination data for a specific patient.

    Args:
        pe_request (PhysicalExamRequest): The physical examination request object.
        patient_data (pd.DataFrame): The DataFrame containing history_pe_admedication_diagnosis.
        patient_id (str): The ID of the patient.

    Returns:
        Optional[str]: The physical examination data or None if not available.
    """

    if patient_data.empty:
        print(
            f"Physical examination data is missing for patient_id: {patient_id}"
        )
        return None

    pe_data = patient_data.iloc[0].get("pe", None)

    if pe_data is None or pd.isna(pe_data):
        print(f"Physical examination data is missing for patient_id: {patient_id}")
        return None

    pe_data = pe_data.strip()


    return pe_data


def fetch_microbiology_results(
    patient_data: pd.DataFrame, 
    patient_id: str, 
    patient_hadm_id: str, 
    microbiology_request: MicrobiologyRequest
) -> Dict[str, Any]:
    """
    Fetches a microbiology result by test name, based on simplified pkl structure.

    Args:
        patient_data (pd.DataFrame): Microbiology data with 'test_name' and 'grouped_microbio_str'.
        patient_id (str): Patient ID
        patient_hadm_id (str): Hospital admission ID
        microbiology_request (MicrobiologyRequest): Request containing 'microbiology_value' as test_name

    Returns:
        Optional[Dict[str, Any]]: Dict with test_name and result, or None if not found.
    """

    if patient_data.empty:
        print(f"Microbiology data is missing for patient_id: {patient_id}")
        return None

    test_name = microbiology_request.microbiology_value  # e.g., "URINE CULTURE"

    filtered = patient_data[patient_data["test_name"] == test_name]
    # print('patient_data:', patient_data)

    if filtered.empty:
        print(f"[Microbiology] No results for {test_name} in patient {patient_id}")
        return None

    result = filtered.iloc[0]

    return result


def fetch_radiology_results(
    patient_data: pd.DataFrame,
    patient_id: str,
    patient_hadm_id: str | int,
    radiology_request: RadiologyRequest,
) -> Optional[pd.Series]:
    """
    Fetches the radiology report for a specific patient and a requested modality and region.
    Returns the first matching report (no timestamp assumed in pkl-based data).
    """

    if patient_data.empty:
        print(f"Radiology data is missing for patient_id: {patient_id}")
        return None

    modality = radiology_request.modality
    region = radiology_request.region

    # Manual correction logic (if needed)
    if modality == "CT" and region == "Venous":
        region = "Chest"
        print(f"[Fix] Changed region to Chest for patient {patient_hadm_id} (Venous CT)")

    if patient_hadm_id in [23794159, 25868499] and modality == "CT" and region == "Abdomen":
        modality = "CTU"
        print(f"[Fix] Changed modality to CTU for patient {patient_hadm_id}")

    # Apply filter
    filtered_data = patient_data[
        (patient_data["modality"] == modality) & (patient_data["region"] == region)
    ]

    if filtered_data.empty:
        print(f"No radiology reports found for patient_id: {patient_id}, modality: {modality}, region: {region}")
        return None

    result = filtered_data.iloc[0]

    return result


def fetch_procedure_search_results(
    collection: Qdrant_Collection,
    patient_id: str,
    patient_hadm_id: str | int,
    procedure_request: ProcedureSearch,
) -> Optional[List[dict]]:
    """
    Fetches the top_k matching procedure codes using vector database search.

    Args:
        procedure_request (ProcedureRequestFHIR): The procedure request object.
        collection (Qdrant_Collection): The vector database collection.

    Returns:
        Optional[List[dict]]: The top_k matching procedure codes and metadata, or None if not found.
    """
    query = procedure_request.procedure
    top_k = 10  # adjust if needed
    procedure_options = collection.search(query, query_filter=None, top_k=top_k)

    if not procedure_options:
        print(f"No procedure codes found for query: {query}")
        return None

    # maybe implement the selection function here

    # print(colored(procedure_options, "cyan"))
    return procedure_options


def fetch_procedure_request_results(
    # collection: Qdrant_Collection,
    patient_id: str,
    patient_hadm_id: str | int,
    procedure_request: ProcedureRequest,
) -> Optional[List[dict]]:
    """
    Returns the procedure request.
    """
    # print(colored(procedure_request.procedure, "cyan"))
    return procedure_request.procedure



def fetch_medication_results(
    patient_data: pd.DataFrame,
    patient_id: str,
    patient_hadm_id: str | int,
    medication_request: MedicationRequest,
) -> Optional[pd.Series]:
    """
    Simulates fetching medication results for a specific patient and medication.

    Args:
        medication_request (MedicationRequest): The medication request object.
        patient_data (pd.DataFrame): The DataFrame containing patient medication events (not used here).
        patient_id (str): The ID of the patient.

    Returns:
        Optional[pd.Series]: Simulated confirmation data or None if simulation fails.
    """
    _, _ = patient_data, patient_id  # ignore these for now

    print(
        "Calling `fetch_medication_results` is a placeholder for returning the requested medications, not any ground truth from the MIMIC dataset."
    )

    simulated_result = pd.Series(
        {
            "drug_name": medication_request.drug_name,
            "dosage_text": medication_request.dosage_text,
            "dosage_value": medication_request.dosage_value,
            "dosage_unit": medication_request.dosage_unit,
            "period": medication_request.period,
            "period_unit": medication_request.period_unit,
            "frequency": medication_request.frequency,
            "route": medication_request.route.value,
            "issued": datetime.now().isoformat(),
        }
    )

    return simulated_result

    