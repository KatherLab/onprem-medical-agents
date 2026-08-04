import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..MimicEnums import *
from .tool_request import *
from ..qdrant_collection import Qdrant_Collection



def fetch_lab_results(patient_data: pd.DataFrame, patient_id: str, patient_hadm_id: str, lab_request: LabRequest) -> Dict[str, Any]:
    """
    Retrieves blood lab test results for a specific patient and test type.

    This function searches the patient's lab event data to find the requested test results
    related to blood tests. It returns the most recent result within the hospital stay.

    Args:
        patient_data (pd.DataFrame): The DataFrame containing patient lab test data.
        patient_id (str): The unique identifier for the patient.
        patient_hadm_id (str): The unique identifier for the patient's hospital admission.
        request (LabRequest): The request object containing a valid lab test name.

    Returns:
        Dict[str, Any]: A dictionary containing the lab test result with:
            - "charttime" (str): Timestamp of the test.
            - "value" (Any): The measured value of the test.
            - "unit" (str): The unit of measurement (if available).
            - "label" (str): The name of the test.
        If no matching result is found, returns an empty dictionary `{}`.
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

    # Ensure "charttime" is in datetime format
    filtered.loc[:, "charttime"] = pd.to_datetime(filtered["charttime"])

    # Restrict to results within the first 24 hours of admission
    filtered = filtered[
        filtered["charttime"] < (filtered["charttime"].min() + pd.Timedelta(days=1))
    ]

    # Return the earliest available test result in dictionary format
    result = filtered.sort_values(by="charttime", ascending=True).iloc[0]
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

    # Ensure "charttime" is in datetime format
    filtered.loc[:, "charttime"] = pd.to_datetime(filtered["charttime"])

    # Restrict to results within the first 24 hours of admission
    filtered = filtered[
        filtered["charttime"] < (filtered["charttime"].min() + pd.Timedelta(days=1))
    ]

    # Return the earliest available test result in dictionary format
    result = filtered.sort_values(by="charttime", ascending=True).iloc[0]
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
    Fetches the microbiology results for a specific patient and a requested microbiology test.
    Returns the first (in time order) test and result that was conducted.

    Args:
        patient_data (pd.DataFrame): The DataFrame containing microbiology data.
        patient_id (str): The ID of the patient.

    Returns:
        Optional[pd.DataFrame]: The microbiology results or None if not available.
    """

    if patient_data.empty:
        print(f"Microbiology data is missing for patient_id: {patient_id}")
        return None

    microbiology_value =microbiology_request.microbiology_value
    patient_data = patient_data.loc[
        patient_data["test_name"] == microbiology_value
    ]
    # print('patient_data:', patient_data)

    if patient_data.empty:
        print(
            f"No microbiology results found for patient_id: {patient_id} and microbiology_value: {microbiology_value}.\n"
            "Returning 'None' that will be returned to the LLM as a missing microbiology result."
        )
        return None

    # Convert 'charttime' to datetime and sort
    patient_data = patient_data.copy()  # prevent SettingWithCopyWarning :D
    patient_data.loc[:, "charttime"] = pd.to_datetime(
        patient_data["charttime"], errors="coerce"
    )
    patient_data.sort_values(by="charttime", ascending=True, inplace=True)
    # print('patient_data:', patient_data)

    # Get the first 'micro_specimen_id' after sorting
    micro_specimen_id = patient_data.iloc[0]["micro_specimen_id"]
    if micro_specimen_id is None:
        print(f"Micro specimen ID is missing for patient_id: {patient_id}")
        return None

    # Create 'microbio_subset' based on 'micro_specimen_id'
    microbio_subset = patient_data.loc[
        patient_data["micro_specimen_id"] == micro_specimen_id
    ].copy()

    if microbio_subset.empty:
        print(
            f"No microbiology subset found for patient_id: {patient_id} and micro_specimen_id: {micro_specimen_id}"
        )
        return None

    # Convert 'storetime' to datetime and sort
    microbio_subset["storetime"] = pd.to_datetime(
        microbio_subset["storetime"], errors="coerce"
    )
    microbio_subset.sort_values(by="storetime", ascending=True, inplace=True)
    # print('microbio_subset:', microbio_subset)

    return microbio_subset


def fetch_radiology_results(
    patient_data: pd.DataFrame,
    patient_id: str,
    patient_hadm_id: str | int,
    radiology_request: RadiologyRequest,
) -> Optional[pd.Series]:
    """
    Fetches the radiology report for a specific patient and a requested modality and region.
    Returns the first (in time order) report that matches the modality and region.

    Args:
        radiology_request (RadiologyRequest): The radiology request object.
        patient_data (pd.DataFrame): The DataFrame containing radiology data.
        patient_id (str): The ID of the patient.
        patient_hadm_id (str | int): The hadm id of the patient.

    Returns:
        Optional[pd.Series]: The radiology report or None if not available.
    """
    # Assuming that patient_data has a DataFrame with radiology reports
    # Let's assume it's in patient_data.radiology_reports
    # radiology_data = patient_data.radiology

    if patient_data.empty:
        print(f"Radiology data is missing for patient_id: {patient_id}")
        return None

    # Filter by modality and region
    modality = radiology_request.modality
    print('modality:', modality)
    region = radiology_request.region
    print('region:', region)

    if modality == "CT" and region == "Venous":
        region = "Chest"
        print(
            colored(
                f"Manually fixing region for patient_hadm_id: {patient_hadm_id}",
                "red",
            )
        )

    # manually fix the modality for these two patients
    if patient_hadm_id in [
        23794159,
        25868499,
    ]:  # both of them have CTU instead of CT Abdomen
        if modality == "CT" and region == "Abdomen":
            print(
                colored(
                    f"Manually fixing modality for patient_hadm_id: {patient_hadm_id}",
                    "red",
                )
            )
            modality = "CTU"

    # Filter the radiology_data DataFrame
    filtered_data = patient_data[
        (patient_data["modality"] == modality) & (patient_data["region"] == region)
    ]

    if filtered_data.empty:
        print(
            f"No radiology reports found for patient_id: {patient_id}, modality: {modality}, and region: {region}.\n"
            "Returning 'None' that will be returned to the LLM as a missing radiology report."
        )

        return None

    # Sort by date and get the first report
    filtered_data = filtered_data.sort_values(by="charttime", ascending=True)
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
            # "route": medication_request.route.value,
            "route": medication_request.route,
            "issued": datetime.now().isoformat(),
        }
    )

    return simulated_result



def lab_aidoc(patient_data: pd.DataFrame, patient_id: str, patient_hadm_id: str, lab_request: LabRequest):

    filtered = patient_data[
        (patient_data["label"] == lab_request.lab_value)
        & (patient_data["fluid"] == "Blood")
    ]

    if filtered.empty:
        print(
            f"No lab results found for patient_id: {patient_id} and lab_value: {lab_request.lab_value}.\n"
            "Returning 'None' that will be returned to the LLM as a missing lab result."
        )
        return None

    # Ensure "charttime" is a datetime column
    filtered.loc[:, "charttime"] = pd.to_datetime(filtered["charttime"])

    filtered = filtered[
        filtered["charttime"] < (filtered["charttime"].min() + pd.Timedelta(days=1))
    ]

    # We just return the earliest lab value result that is available for this hospital admission
    result = filtered.sort_values(by="charttime", ascending=True).iloc[0]
    return result



if __name__ == '__main__':
    from dataset.mimic_dataset import MIMIC_Dataset

    ds=MIMIC_Dataset.load_dataset("pancreatitis")
    # print('ds:', ds)
    # print('ds length:', len(ds))
    patient_data = ds[22412352].lab_events

    hadm_id = patient_data.admissions.hadm_id.values[0]
    print('hadm_id:', hadm_id)

    patient_id = str(patient_data.patients.subject_id.values[0])
    print('patient_id:', patient_id)

    request=LabRequest(lab_value="White Blood Cells")
    print('request:', request)
    lab_results = fetch_lab_results(patient_data=patient_data, patient_id=patient_id, patient_hadm_id=hadm_id, lab_request=request)
    print('fetch_lab_results lab_results:', lab_results)

    lab_results_cmd = lab_aidoc(patient_data=patient_data, patient_id=patient_id, patient_hadm_id=hadm_id, lab_request=request)
    print('cmd_results lab_results:', lab_results)




    # request=UrineRequest(urine_value="Leukocytes")
    # print('request:', request)
    # urine_results = fetch_urine_results(patient_data=patient_data, patient_id=patient_id, patient_hadm_id=hadm_id,request=request)
    # print('lab_results:', urine_results)

    # request=PhysicalExamRequest()
    # print('request:', request)
    # pe_results = fetch_pe_results(patient_data=patient_data, patient_id=patient_id, patient_hadm_id=hadm_id)
    # print('lab_results:', pe_results)
    
    # request=MicrobiologyRequest(microbiology_value="URINE CULTURE")
    # print('request:', request)
    # microbiology_results = fetch_microbiology_results(patient_data=patient_data, patient_id=patient_id, patient_hadm_id=hadm_id,request=request)
    # print('lab_results:', microbiology_results)

    # request=RadiologyRequest(modality="CT", region="Abdomen")
    # print('request:', request)
    # radiolog_results = fetch_radiology_results(patient_data=patient_data, patient_id=patient_id, patient_hadm_id=hadm_id,radiology_request=request)
    # print('lab_results:', radiolog_results)
    