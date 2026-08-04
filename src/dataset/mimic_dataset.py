# =================================================
# Codes are adapted from MIRA
# =================================================

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm
import pickle

from ..MimicEnums import *

class MIMIC_Dataset(BaseModel):
    """
    Class to hold data for a MIMIC dataset filtered for a specific diagnosis.

    This class represents a comprehensive dataset from the MIMIC database, filtered for a particular diagnosis.
    It contains various dataframes representing different aspects of patient care and hospital stays.

    Attributes:
        diagnosis (str): The specific diagnosis this dataset is filtered for.
        hadm_ids (set[int]): A set of unique hospital admission IDs ('hadm_id') for patients with the specified diagnosis.
        lab_events (pd.DataFrame): Laboratory test results for the filtered patients.
        microbiology (pd.DataFrame): Microbiology test results for the filtered patients.
        history_pe_admedication_diagnosis (pd.DataFrame): Patient history, physical examination, admission medication, and diagnosis extracted from discharge letters.
        radiology (pd.DataFrame): Radiology reports for the filtered patients.
        medication (pd.DataFrame): Medication prescriptions during hospital stays.
        procedures_icd (pd.DataFrame): ICD codes and descriptions for procedures performed during hospital stays.
        diagnosis_icd (pd.DataFrame): ICD diagnoses for the hospital stays.
        diagnosis_ed (pd.DataFrame): Emergency department diagnoses.
        ed_stays (pd.DataFrame): Information about emergency department stays.
        medrecon (pd.DataFrame): Medication reconciliation data from the emergency department.
        pyxis (pd.DataFrame): Medication administration data from the Pyxis system in the emergency department.
        triage (pd.DataFrame): Triage information from the emergency department.
        vitalsign (pd.DataFrame): Continuous vital sign measurements from the emergency department.
        admissions (pd.DataFrame): Admissions data for the filtered patients.
        patients (pd.DataFrame): Patients data for the filtered patients.

    The class is iterable, allowing iteration over individual hospital admissions (hadm_ids).
    Each iteration returns a Mimic_Hadm_Dataset object containing data for a single hospital admission.

    Methods:
        __iter__: Allows iteration over the dataset's hospital admissions.
        __next__: Returns the next Mimic_Hadm_Dataset in the iteration.

    Usage:
        mimic_dataset = Mimic_Dataset(diagnosis="appendicitis", ...)
        for single_admission in mimic_dataset:
            # Process each admission
    """

    diagnosis: str = Field(
        ..., description="The diagnosis this dataset is filtered for"
    )
    hadm_ids: set[int] = Field(
        ..., description="A set of 'hadm_id's of the patients filtered for a diagnosis"
    )

    lab_events: pd.DataFrame = Field(..., description="The lab events for the patients")
    history_pe_admedication_diagnosis: pd.DataFrame = Field(
        ...,
        description="The history, physical examination (pe), admission medication (admedication) and diagnosis for the patients, extracted from the discharge letters",
    )
    microbiology: pd.DataFrame = Field(
        ..., description="The microbiology test results for the patients"
    )
    radiology: pd.DataFrame = Field(
        ..., description="The radiology reports for the patients"
    )
    medication: pd.DataFrame = Field(
        ...,
        description="The medication (prescriptions) for the patients during the hospital stay",
    )
    procedures_icd: pd.DataFrame = Field(
        ...,
        description="ICD codes and descriptions for the procedures performed on the patients during hospital stay",
    )
    diagnosis_icd: pd.DataFrame = Field(
        ..., description="The diagnosis_icd for the patients as per hospital stay"
    )
    diagnosis_ed: pd.DataFrame = Field(
        ...,
        description="The diagnosis_ed for the patients from the emergency department",
    )
    ed_stays: pd.DataFrame = Field(..., description="The ed_stays for the patients")
    medrecon: pd.DataFrame = Field(
        ..., description="The medication for the patients in the emergency department"
    )
    pyxis: pd.DataFrame = Field(
        ..., description="The medication administration in the mergency departmen"
    )
    triage: pd.DataFrame = Field(
        ..., description="The triage for the patients in the emergency department"
    )
    vitalsign: pd.DataFrame = Field(
        ...,
        description="The continuous measurements of vitalsigns for the patients in the emergency department",
    )
    admissions: pd.DataFrame = Field(
        ..., description="The admissions data for the patients"
    )
    patients: pd.DataFrame = Field(
        ..., description="The patients data for the patients"
    )

    class Config:
        arbitrary_types_allowed = True  # allow pd.DataFrames as types
        populate_by_name = True

    def list_tables(self):
        return [
            attr
            for attr in self.__dict__.keys()
            if isinstance(getattr(self, attr), pd.DataFrame)
        ]

    def __getitem__(self, idx):
        if idx not in self.hadm_ids:
            raise KeyError(f"'hadm_id' {idx} not found in the dataset.")

        df_kwargs = {}
        for attr in self.list_tables():
            df = getattr(self, attr)
            assert "hadm_id" in df.columns, f"Column 'hadm_id' not found in {attr}."
            assert df["hadm_id"].dtype in [
                int,
                float,
            ], f"Column 'hadm_id' in {attr} is of type {df['hadm_id'].dtype} instead of int."
            df_filtered = df[df["hadm_id"] == idx].copy()
            df_kwargs[attr] = df_filtered

        # Remove 'hadm_ids' from kwargs if it's accidentally included
        df_kwargs.pop("hadm_ids", None)

        return MIMIC_Hadm_Dataset(
            diagnosis=self.diagnosis,
            hadm_ids=self.hadm_ids,
            hadm_id=idx,
            **df_kwargs,
        )

    def __len__(self):
        return len(self.hadm_ids)

    def __iter__(self):
        # make the class iterable over the hadm_ids
        self._iter_hadm_ids = iter(self.hadm_ids)
        return self

    def __next__(self):
        # get the next item until exhausted
        try:
            hadm_id = next(self._iter_hadm_ids)

        except StopIteration:
            raise StopIteration

        return MIMIC_Hadm_Dataset(
            diagnosis=self.diagnosis,
            hadm_ids=self.hadm_ids,  # will be removed after passing to super().__init__
            hadm_id=hadm_id,
            lab_events=self.lab_events[self.lab_events.hadm_id == hadm_id],
            microbiology=self.microbiology[self.microbiology.hadm_id == hadm_id],
            history_pe_admedication_diagnosis=self.history_pe_admedication_diagnosis[
                self.history_pe_admedication_diagnosis.hadm_id == hadm_id
            ],
            radiology=self.radiology[self.radiology.hadm_id == hadm_id],
            medication=self.medication[self.medication.hadm_id == hadm_id],
            procedures_icd=self.procedures_icd[self.procedures_icd.hadm_id == hadm_id],
            diagnosis_icd=self.diagnosis_icd[self.diagnosis_icd.hadm_id == hadm_id],
            diagnosis_ed=self.diagnosis_ed[self.diagnosis_ed.hadm_id == hadm_id],
            ed_stays=self.ed_stays[self.ed_stays.hadm_id == hadm_id],
            medrecon=self.medrecon[self.medrecon.hadm_id == hadm_id],
            pyxis=self.pyxis[self.pyxis.hadm_id == hadm_id],
            triage=self.triage[self.triage.hadm_id == hadm_id],
            vitalsign=self.vitalsign[self.vitalsign.hadm_id == hadm_id],
            admissions=self.admissions[self.admissions.hadm_id == hadm_id],
            patients=self.patients[self.patients.hadm_id == hadm_id],
        )

    def save_dataset(self, name: str = None):
        """Save the dataset to a metadata.json and parquet (pd.DataFrame) files"""
        if name is None:
            name = self.diagnosis
        path = Path(
            f"/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/dataset_test/{name}"
        )
        if path.exists():
            user_input = input(
                f"The path {path} already exists. Do you want to force overwrite? (y/n): "
            )
            if user_input.lower() != "y":
                print("Operation cancelled by user.")
                return

        path.mkdir(parents=True, exist_ok=True)
        metadata = {"diagnosis": self.diagnosis, "hadm_ids": list(self.hadm_ids)}

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        for attr_name, attr_value in tqdm(self.__dict__.items()):
            if isinstance(attr_value, pd.DataFrame):
                # handle mixed types which break in .parquet files
                # if one column has more than 1 type, convert it to string
                for col in attr_value.select_dtypes(include=["object"]).columns:
                    if attr_value[col].apply(type).nunique() > 1:
                        print(
                            f"Column '{col}' in DataFrame '{attr_name}' has mixed types. Converting to string for .parquet storage."
                        )
                        attr_value[col] = (
                            attr_value[col].astype(str).replace("nan", pd.NA)
                        )

                attr_value.to_parquet(path / f"{attr_name}.parquet")

        print(f"Saved dataset for {name} to {path}.")

    @classmethod
    def load_dataset(cls, diagnosis_name: str) -> MIMIC_Dataset:
        """Load the dataset from a metadata.json and parquet (pd.DataFrame) files"""
        path = Path(
            f"data/{diagnosis_name}"
        )

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        diagnosis, hadm_ids = metadata["diagnosis"], set(metadata["hadm_ids"])

        dataframes = {}
        for filename in tqdm(path.glob("*.parquet")):
            # print('filename:', filename)
            df_name = filename.stem
            # print('df_name:', df_name)
            dataframes[df_name] = pd.read_parquet(filename)

        # print(f"Loaded dataset for {name} from {path}.")

        return cls(diagnosis=diagnosis, hadm_ids=hadm_ids, **dataframes)

    @classmethod
    def load_from_pkl(cls, diagnosis_name: str) -> "MIMIC_Dataset":
        with open(f"/mnt/bulk-vega/lizhang/LiWS/medical_bench/datasets/mimic_cdm/physionet.org/files/mimic-iv-ext-cdm/1.1/{diagnosis_name}_hadm_info_first_diag.pkl", "rb") as f:
            data_dict = pickle.load(f)

        hadm_ids = set(data_dict.keys())

        # Load lab_test_mapping.csv
        mapping_df = pd.read_csv('/mnt/bulk-vega/lizhang/LiWS/medical_bench/datasets/mimic_cdm/physionet.org/files/mimic-iv-ext-cdm/1.1/lab_test_mapping.csv')

        # Build mapping dicts: itemid -> label/fluid.
        itemid_to_label = mapping_df.set_index('itemid')['label'].to_dict()
        itemid_to_fluid = mapping_df.set_index('itemid')['fluid'].to_dict()

        itemid_to_test_name = {
            int(k[1:]): v.value for k, v in MicroBiologyValue.__members__.items()
        }

        # Initialize empty lists for each medical data type.
        admissions_data = []
        patients_data = []
        history_pe_admedication_diagnosis_data = []
        lab_events_data = []
        microbiology_data = []
        radiology_data = []
        procedures_icd_data = []
        diagnosis_icd_data = []
        # medication_data = []
        # diagnosis_ed_data = []  
        # ed_stays_data = []    
        # medrecon_data = []      
        # pyxis_data = []         
        # triage_data = []        
        # vitalsign_data = [] 
        empty_int_df = pd.DataFrame({"hadm_id": pd.Series([], dtype=int)})

        for hadm_id, data in data_dict.items():
            # History and Physical Examination
            history_pe_admedication_diagnosis_data.append({
                "hadm_id": hadm_id,
                "extracted_history": data.get("Patient History", ""),
                "pe": data.get("Physical Examination", ""),
                "admission_medication": "",  # placeholder if not present in data
                "discharge_diagnosis_from_text": data.get("Discharge Diagnosis", "")
            })

            # Laboratory tests
            for itemid, value in data.get("Laboratory Tests", {}).items():
                # Get reference range lower/upper.
                ref_lower = data.get("Reference Range Lower", {}).get(itemid, None)
                ref_upper = data.get("Reference Range Upper", {}).get(itemid, None)
                
                # Get label and fluid info.
                label = itemid_to_label.get(itemid, "Unknown")
                fluid = itemid_to_fluid.get(itemid, "Unknown")
                
                # Append to lab_events_data.
                lab_events_data.append({
                    "hadm_id": hadm_id,
                    "itemid": itemid,
                    "value": value,
                    "ref_range_lower": ref_lower,
                    "ref_range_upper": ref_upper,
                    "label": label,
                    "fluid": fluid
                })

            # Microbiology tests
            for itemid, value in data.get("Microbiology", {}).items():
                microbiology_data.append({
                    "hadm_id": hadm_id,
                    "test_itemid": itemid,
                    "test_name": itemid_to_test_name.get(itemid, "Unknown Test"),
                    "grouped_microbio_str": value
                })

            # Radiology reports
            for rad_report in data.get("Radiology", []):
                radiology_data.append({
                    "hadm_id": hadm_id,
                    "modality": rad_report["Modality"],
                    "region": rad_report["Region"],
                    "extracted_rad_events": rad_report["Report"]
                })

            # ICD diagnoses
            for diag in data.get("ICD Diagnosis", []):
                diagnosis_icd_data.append({
                    "hadm_id": hadm_id,
                    "long_title": diag
                }) # TO_DO:

            # ICD procedures
            for proc_code, proc_title in zip(data.get("Procedures ICD9", []), data.get("Procedures ICD9 Title", [])):
                procedures_icd_data.append({
                    "hadm_id": hadm_id,
                    "icd_code": proc_code,
                    "long_title": proc_title
                }) # TO_DO:

            # Admission and patient data can be placeholders.
            admissions_data.append({
                "hadm_id": hadm_id,
                "subject_id": f"subject_{hadm_id}"
            })

            patients_data.append({
                "subject_id": f"subject_{hadm_id}",
                "gender": "UNKNOWN",  # fill if available
                "dob": None,  # fill if available
                "hadm_id": hadm_id
            })

        return cls(
            diagnosis=diagnosis_name,
            hadm_ids=hadm_ids,
            lab_events=pd.DataFrame(lab_events_data),
            microbiology=pd.DataFrame(microbiology_data),
            history_pe_admedication_diagnosis=pd.DataFrame(history_pe_admedication_diagnosis_data),
            radiology=pd.DataFrame(radiology_data),
            medication=empty_int_df.copy(), 
            procedures_icd=pd.DataFrame(procedures_icd_data),
            diagnosis_icd=pd.DataFrame(diagnosis_icd_data),
            diagnosis_ed=empty_int_df.copy(),  
            ed_stays=empty_int_df.copy(),      
            medrecon=empty_int_df.copy(),      
            pyxis=empty_int_df.copy(),         
            triage=empty_int_df.copy(),        
            vitalsign=empty_int_df.copy(),     
            admissions=pd.DataFrame(admissions_data),
            patients=pd.DataFrame(patients_data)
        )



class MIMIC_Hadm_Dataset(MIMIC_Dataset):
    """
    Class to hold data for a single MIMIC hospital admission.

    This class represents a dataset for an individual hospital admission in the MIMIC database.
    It inherits from Mimic_Dataset and contains specific data related to a single hospital
    admission identified by a unique 'hadm_id'.

    Attributes:
        hadm_id (int): The unique identifier for the hospital admission.
        lab_events (pd.DataFrame): Laboratory test results for this admission.
        discharge_text (pd.DataFrame): Discharge summary text for this admission.
        radiology (pd.DataFrame): Radiology reports for this admission.
        medication (pd.DataFrame): Medication prescriptions for this admission.
        microbiology (pd.DataFrame): Microbiology test results for this admission.
        procedures_icd (pd.DataFrame): ICD procedure codes for this admission.
        diagnosis_icd (pd.DataFrame): ICD diagnoses for this admission.
        diagnosis_ed (pd.DataFrame): Emergency department diagnoses for this admission.
        ed_stays (pd.DataFrame): Emergency department stay information for this admission.
        medrecon (pd.DataFrame): Medication reconciliation data for this admission.
        pyxis (pd.DataFrame): Medication administration data from Pyxis for this admission.
        triage (pd.DataFrame): Triage information for this admission.
        vitalsign (pd.DataFrame): Vital sign measurements for this admission.

    The class provides methods to access and manipulate data specific to a single
    hospital admission, facilitating analysis and processing of individual patient stays.
    """

    hadm_id: int = Field(
        ..., description="The 'hadm_id' of the specific hospital admission"
    )

    class Config:
        arbitrary_types_allowed = True  # to allow pd.DataFrame's

    def __init__(self, **data):
        """Remove the hadm_ids attribute from the dataset"""
        super().__init__(**data)
        if hasattr(self, "hadm_ids"):
            del self.hadm_ids

    def __repr__(self):
        return f"MIMIC_Hadm_Dataset(hadm_id={self.hadm_id})"

    def __str__(self):
        dataframes = [
            (name, field.description)
            for name, field in self.model_fields.items()
            if isinstance(field.annotation, type(pd.DataFrame))
        ]

        output = [f"MIMIC Hospital Admission Dataset (hadm_id: {self.hadm_id})"]
        output.append("Dataframes:")
        for name, description in dataframes:
            output.append(f"  - {name}: {description}")

        return "\n".join(output)



if __name__ == '__main__':
    
    SELECTED_HADM_IDS: Dict[str, List[int]] = {
        "appendicitis": [
            21520386,
            29935618,
            22235144,
            23136283,
            29829147,
            29653022,
            29786142,
            22644768,
            22763551,
            21489708,
            23754804,
            29866037,
            25428023,
            26615866,
            20174907,
            22210621,
            26853438,
            21987392,
            26859594,
            23066698,
            27428942,
            27588689,
            21983333,
            24971365,
            22771821,
            21479535,
            28231795,
            24498295,
            25075835,
            21477504,
            22773895,
            24438918,
            21473416,
            23105675,
            22519948,
            28119182,
            29466775,
            27676824,
            28962968,
            25843876,
            26202278,
            21448873,
            23603369,
            29255851,
            27433131,
            26374326,
            22757559,
            29335735,
            27220150,
            20095162,
            25696441,
            21031107,
            24694980,
            26300621,
            22048977,
            24342739,
            26593491,
            25868499,
            23838935,
            28412125,
            27738338,
            29630691,
            27646180,
            21074150,
            25579760,
            23288051,
            25671925,
            20705529,
            23556345,
            26441980,
            26292478,
            26495236,
            21229829,
            24529163,
            23277838,
            25188622,
            21782802,
            20152597,
            29806870,
            26642709,
            21039382,
            29079831,
            25172248,
            25045274,
            26269975,
            24887581,
            25905440,
            23101737,
            25733420,
            25053495,
            28123458,
            25542984,
            25190734,
            23044437,
            24955222,
            26485080,
            25731420,
            29403489,
            22618470,
            29684076,
            20312429,
            29309294,
            27890040,
            28205438,
            20345216,
            26184067,
            20486536,
            23761295,
            21997969,
            22581653,
            20890008,
            22487450,
            20275614,
            27107744,
            20883895,
            29282744,
            23706042,
            25635259,
            23372228,
            26481091,
            21516740,
            20679109,
            24009162,
            27562444,
            28164567,
            26436057,
            20046307,
            25438691,
            22852072,
            22852072,
            22559214,
            29725167,
            23794159,
            26302960,
            21596659,
            26405365,
            29592061,
            22594046,
            27879943,
            22663693,
            24140304,
            22200862,
            24257060,
            23067173,
            29590055,
            23853611,
            29776429,
            28496435,
            24513076,
            25604665,
        ],
        "asthma": [
            20222085,
            25665542,
            27558149,
            26141190,
            27516167,
            22177543,
            23977865,
            26383244,
            22154509,
            27851540,
            27911317,
            22086293,
            26836505,
            20410780,
            28286878,
            26927655,
            27568426,
            28142639,
            21303599,
            24501679,
            29953459,
            24694324,
            26862256,
            22177078,
            24868784,
            25339576,
            21483441,
            26249916,
            29901501,
            21375936,
            27469634,
            26391490,
            29527623,
            24923079,
            29970251,
            21664471,
            25208155,
            27982941,
            28602718,
            22784862,
            27688545,
            23306086,
            27204327,
            28831978,
            25521392,
            24917364,
            26763379,
            29365108,
            29881460,
            26476661,
            22754680,
            26174710,
            27084410,
            23051003,
            21941756,
            21512565,
        ],
        "cholecystitis": [
            21210624,
            20226562,
            20185611,
            23538206,
            27194914,
            23051304,
            23219752,
            21712427,
            26810924,
            20275759,
            29850672,
            20578869,
            28477496,
            28892217,
            21060670,
            29600831,
            28627522,
            26927175,
            26402893,
            21976655,
            24534610,
            26618451,
            23444567,
            24593504,
            27071073,
            29090915,
            26999909,
            25039466,
            21707885,
            28337776,
            22393969,
            28327537,
            20772979,
            23545972,
            29913717,
            21888122,
            25527423,
            27714689,
            25092228,
            24140420,
            27673222,
            28456075,
            20606097,
            20121749,
            25140885,
            27948194,
            29238948,
            20564135,
            24789161,
            28726445,
            23768237,
            27934384,
            22596275,
            20012726,
            22054590,
            25483967,
            21636805,
            29116104,
            28697802,
            26076363,
            23898846,
            27621604,
            29250795,
            22084846,
            24713967,
            20760817,
            29300977,
            23555318,
            24456954,
            22829311,
            28276491,
            21219598,
            23169808,
            24056597,
            23838497,
            27670821,
            28614442,
            24771883,
            29634349,
            28292400,
            21558071,
            23968056,
            21007672,
            25552186,
            26332987,
            29060924,
            27426621,
            23625534,
            22175558,
            20771143,
            26573641,
            29565262,
            24714062,
            26286928,
            25366350,
            23352147,
            23649113,
            28555100,
            24601951,
            25527137,
            28243297,
            24243552,
            28809063,
            20875625,
            28015467,
            24037231,
            22049139,
            29283190,
            21049206,
            27554689,
            20723074,
            21555075,
            28973441,
            28503432,
            28177296,
            29030289,
            23550354,
            26157459,
            29709207,
            20520856,
            23313820,
            23383965,
            21602208,
            21273009,
            23591858,
            27524531,
            27770296,
            25360313,
            29055421,
            20121534,
            21944769,
            22525891,
            29656006,
            22080966,
            22221259,
            25916880,
            29697493,
            24078805,
            25315798,
            25194454,
            22969824,
            24466918,
            22332907,
            26543596,
            29044720,
            25574385,
            28066809,
            24908797,
            26549810,
            28379853,
        ],
        "diverticulitis": [
            29381122,
            26292867,
            # 27323780,
            # 27737351,
            # 26869001,
            # 28678157,
            # 20085778,
            # 29270681,
            # 27923482,
            # 29717153,
            # 26007074,
            # 20817953,
            # 20287524,
            # 21037733,
            # 23347623,
            # 24244647,
            # 25216936,
            # 29137700,
            # 29495801,
            # 27411121,
            # 25194934,
            # 26581302,
            # 28709561,
            # 20627898,
            # 27789625,
            # 24736188,
            # 27563327,
            # 28496070,
            # 26053830,
            # 26360008,
            # 22607946,
            # 27541971,
            # 21439187,
            # 27644374,
            # 21177686,
            # 23546579,
            # 25841363,
            # 24642650,
            # 27173471,
            # 21055588,
            # 27918309,
            # 28131817,
            # 20348908,
            # 20009581,
            # 28132333,
            # 28260336,
            # 24972913,
            # 22468974,
            # 23525872,
            # 29928181,
            # 27417207,
            # 28694648,
            # 26137978,
            # 24528124,
        ],
        "pancreatic_cancer": [
            24467968,
            27101188,
            22517252,
            20576647,
            26478602,
            28526602,
            24747536,
            28739090,
            22225430,
            21156123,
            22654751,
            21172648,
            22726184,
            27500458,
            29275436,
            27670835,
            23686979,
            27594182,
            28528970,
            20803534,
            22217315,
            20203766,
            26556670,
        ],
        "pancreatitis": [
            28079494,
            24564491,
            28220940,
            24571788,
            20464014,
            24077199,
            21775506,
            29618198,
            28165275,
            24133787,
            27754268,
            21778589,
            21937314,
            23511843,
            25519274,
            22195501,
            26837038,
            29329582,
            29616816,
            27625395,
            28319924,
            28002741,
            23430198,
            24763579,
            27787324,
            24829628,
            27833918,
            21703487,
            22412352,
            28122817,
            28970050,
            29137091,
            29013063,
            21639112,
            21544012,
            20005465,
            28559961,
            27611614,
            28422368,
            21010528,
            24490340,
            20876774,
            26042606,
            24994288,
            29455216,
            29165296,
            24050288,
            21832948,
            27067638,
            23183864,
            26379002,
            22946811,
            25628541,
        ],
        "pneumonia": [
            28519937,
            26259844,
            24472324,
            22539785,
            24588564,
            24408469,
            26621818,
            25611551,
            26669344,
            24280096,
            22379303,
            29519145,
            24404778,
            26644523,
            27045550,
            26255538,
            20769717,
            27254709,
            28053052,
            24822590,
            29331775,
            28198208,
            25422401,
            25467841,
            27698124,
            23427406,
            22718164,
            24133332,
            27742300,
            22998881,
            26666470,
            28819431,
            20704876,
            21166960,
            26577777,
            20139642,
        ],
        "pulmonary_embolism": [
            26406913,
            23905801,
            25220111,
            23924755,
            24915989,
            22183958,
            25434133,
            27214878,
            24830502,
            26225704,
            27014187,
            23058988,
            20255277,
            25194029,
            24307764,
            21600822,
            29157436,
            22361674,
            20642379,
            29040202,
            27817040,
            23657050,
            21259358,
            26361446,
            28153964,
            23080044,
            27302513,
            27808392,
            29539977,
            22493838,
            28124816,
            20932242,
            28694686,
            26190496,
            29084834,
            20648107,
            29797038,
            20988595,
            22307008,
            22164166,
            20690126,
            28857555,
            28493014,
            28357347,
            29976815,
            22497012,
            21279989,
            29023991,
            29904630,
            26608378,
            20212987,
            28755713,
            23459074,
            29144836,
            22935311,
            23393556,
            22151475,
            23745843,
            29715251,
            20483890,
            22642500,
            25465679,
            28475217,
            22736731,
            28997468,
            26320221,
            29546846,
            22826848,
            29251950,
            24226671,
            26401137,
            23779699,
            29316468,
            25860484,
            29144457,
            27032972,
            22278047,
            22382502,
            25395630,
            24928690,
            23694773,
            29500350,
            23453121,
            27960260,
            22505930,
            23314398,
            28685288,
            25335280,
            20559347,
            20895743,
        ],
        "uti": [
            26514952,
            26327048,
            20381197,
            26605588,
            26147860,
            26568726,
            28857881,
            26619930,
            27484714,
            29687339,
            26345516,
            24052287,
            21122114,
            25899074,
            28877892,
            27090501,
            28447307,
            29861967,
            25899098,
            22747235,
            23254131,
            28137592,
            26968697,
            26429561,
            29309564,
            25097347,
            27945105,
            29516435,
            21600416,
            27596450,
            26769570,
            27122343,
            25357480,
            20637355,
            20153014,
            21651639,
            24037580,
            27947724,
            28239571,
            28604130,
            21307639,
            24611575,
            26646795,
            24731925,
            23792919,
            22624032,
            27132196,
            22392635,
            23707964,
            24012613,
            29600070,
            29518662,
            29176655,
            20229970,
            24801622,
            23099734,
            28821849,
            26960224,
            29424494,
            21527409,
            26360178,
        ],
    }


    ds=MIMIC_Dataset.load_dataset("pancreatitis")
    # print('ds:', ds)
    print('ds length:', len(ds))

    # max_samples = 5
    # num_samples = (
    #     min(len(ds.hadm_ids), max_samples) if max_samples else len(ds.hadm_ids)
    # )
    # print('num_samples:', num_samples)
    # print('hadm_ids', ds.hadm_ids)
    hadm_id = 29616816
    patient_data = ds[hadm_id]

    # print('patient_data_label', patient_data.lab_events["label"])
    # print('admissions:', patient_data.admissions)  # inspect admission data
    # hadm_ids = patient_data.admissions.hadm_id.values
    # print('hadm_ids:', hadm_ids) #hadm_ids: [21974897]
    # hadm_id = patient_data.admissions.hadm_id.values#[0] #21974897
    # print('hadm_id.values[0]:', hadm_id)

    print('lab_events:', patient_data.lab_events)  # inspect lab events data
    df = patient_data.lab_events #['label']
    df.to_csv(f'/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/{hadm_id}_lab.csv', index=False, encoding='utf-8')
    # print('admissions:', patient_data.admissions)  # inspect admission data

    # print('history_pe_admedication_diagnosis:', patient_data.history_pe_admedication_diagnosis)  # inspect history data
    df = patient_data.history_pe_admedication_diagnosis
    df.to_csv(f'/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/{hadm_id}_history_pe_admedication_diagnosis.csv', index=False, encoding='utf-8')

    # print('microbiology:', patient_data.microbiology)
    df = patient_data.microbiology
    df.to_csv(f'/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/{hadm_id}_microbiology.csv', index=False, encoding='utf-8')

    # print('radiology:', patient_data.radiology)
    df = patient_data.radiology
    df.to_csv(f'/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/{hadm_id}_radiology.csv', index=False, encoding='utf-8')

    # print('procedures_icd:', patient_data.procedures_icd)
    df = patient_data.procedures_icd
    df.to_csv(f'/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/{hadm_id}_procedures_icd.csv', index=False, encoding='utf-8')

    # print('diagnosis_icd:', patient_data.diagnosis_icd)
    df = patient_data.diagnosis_icd
    df.to_csv(f'/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/{hadm_id}_diagnosis_icd.csv', index=False, encoding='utf-8')

    # print('diagnosis_ed:', patient_data.diagnosis_ed)
    df = patient_data.diagnosis_ed
    df.to_csv(f'/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/{hadm_id}_diagnosis_ed.csv', index=False, encoding='utf-8')

    df = patient_data.triage
    df.to_csv(f'/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/{hadm_id}_triage.csv', index=False, encoding='utf-8')


    # for attr_name, attr_value in patient_data.__dict__.items():
    #     if isinstance(attr_value, pd.DataFrame):
    #         print(f"==== {attr_name} ====")
    #         print(attr_value.columns.tolist())  # print columns as a list
    #         print("\n")

    # import pandas as pd

    # columns_dict = {}
    # for attr_name, attr_value in patient_data.__dict__.items():
    #     if isinstance(attr_value, pd.DataFrame):
    #         columns_dict[attr_name] = attr_value.columns.tolist()

    # df_columns = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in columns_dict.items()]))
    # print(df_columns)

    # from typing import Optional
    # def extract_ground_truth(patient_data) -> Optional[str]:
    #     """
    #     Extracts the most reliable ground truth diagnosis from a patient's records.

    #     Priority:
    #     1. Take the first diagnosis from diagnosis_ed where is_invalid is False.
    #     2. Confirm that the same code exists in diagnosis_icd and is also valid.
    #     """

    #     ed = patient_data.diagnosis_ed
    #     icd = patient_data.diagnosis_icd

    #     # Step 1: filter diagnoses marked valid in ED
    #     valid_ed = ed[ed["is_invalid"] == False]

    #     for _, ed_row in valid_ed.iterrows():
    #         ed_code = ed_row["icd_code"]
    #         ed_version = ed_row["icd_version"]

    #         # Step 2: confirm the diagnosis exists and is valid in the ICD table
    #         match = icd[
    #             (icd["icd_code"] == ed_code)
    #             & (icd["icd_version"] == ed_version)
    #             & (icd["is_invalid"] == False)
    #         ]

    #         if not match.empty:
    #             # Return long_title as ground truth
    #             return match.iloc[0]["long_title"]

    #     return None  # no matching diagnosis
    

    # def extract_ground_truth_new(patient_data) -> Optional[str]:
    #     """
    #     Return the most reliable diagnosis from diagnosis_icd (first valid entry).
    #     """
    #     icd = patient_data.diagnosis_icd

    #     # Filter out invalid diagnoses
    #     valid_diagnoses = icd[icd["is_invalid"] == False]

    #     if not valid_diagnoses.empty:
    #         return valid_diagnoses.iloc[0]["long_title"]

    #     return None
    


    # # Aggregation variables
    # total = 0
    # with_gt = 0
    # without_gt = 0

    # hadm_ids = list(ds.hadm_ids)
    # # for patient_data in tqdm(ds, desc="Processing patients"):
    # for hadm_idx in tqdm(hadm_ids, desc="processing hadm ids "):
    #     patient_data = ds[hadm_idx]
    #     gt = extract_ground_truth(patient_data)

    #     total += 1
    #     if gt is not None:
    #         with_gt += 1
    #     else:
    #         without_gt += 1

    # # Print results
    # print(f"Total patients: {total}")
    # print(f"With ground truth: {with_gt}")
    # print(f"Without ground truth: {without_gt}")
    # print(f"Coverage: {with_gt / total:.2%}")



    # from collections import defaultdict
    # ground_truth_stats = defaultdict(lambda: {"has_gt": 0, "no_gt": 0})

    # for task_name, hadm_id_list in SELECTED_HADM_IDS.items():
    #     for hadm_id in tqdm(hadm_id_list, desc=f"Processing {task_name}"):
    #         if task_name == 'appendicitis':
    #             try:
    #                 patient_data = ds[hadm_id]
    #                 gt = extract_ground_truth_new(patient_data)
    #                 if gt:
    #                     ground_truth_stats[task_name]["has_gt"] += 1
    #                 else:
    #                     print('hadm_id:',hadm_id)
    #                     ground_truth_stats[task_name]["no_gt"] += 1
    #             except KeyError:
    #                 print(f"Warning: hadm_id {hadm_id} not found in dataset")
    #                 ground_truth_stats[task_name]["no_gt"] += 1

    # # Print summary stats
    # print("\n=== Ground Truth Statistics ===")
    # for task, stats in ground_truth_stats.items():
    #     print(f"{task}: {stats['has_gt']} with GT, {stats['no_gt']} without GT")
