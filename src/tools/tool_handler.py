import asyncio
import uuid
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict, Union

import pandas as pd
import logging
import importlib

# from .tool_mimic import *
# from .tool_observation import *
from .tool_request import (
    LabRequest,
    MedicationRequest,
    MicrobiologyRequest,
    PhysicalExamRequest,
    ProcedureRequest,
    RadiologyRequest,
    UrineRequest,
)


logger = logging.getLogger(__name__)


# 1. Import the BENCHMARK setting from the main configuration file.
try:
    from ..configs.agent_config import BENCHMARK
except ImportError:
    # Provide a default for execution in other contexts.
    BENCHMARK = "AIDOC" 
    logger.warning("Could not import BENCHMARK from agent_config, defaulting to 'AIDOC'.")

print(f"INFO: Initializing tool handlers for BENCHMARK: '{BENCHMARK}'")

# 2. Dynamically load the correct implementation module for BENCHMARK.
if BENCHMARK == "AIDOC":
    try:
        # Import from the standard AIDOC paths.
        from .tool_mimic import *
        from .tool_observation import *
        print("  - Successfully imported 'aidoc' tool implementations.")
    except ImportError as e:
        logger.error(f"Failed to import AIDOC tool implementations: {e}")
        raise
elif BENCHMARK == "CDM":
    try:
        # Import from CDM-specific paths.
        from .tool_mimic_cdm import *
        from .tool_observation_cdm import *
        print("  - Successfully imported 'cdm' tool implementations.")
    except ImportError as e:
        logger.error(f"Failed to import CDM tool implementations: {e}")
        raise
elif BENCHMARK == "VIVABENCH":
    # VivaBench tools are implemented in a benchmark-specific module and do not
    # rely on the MIMIC dataframe handlers defined here.
    print("  - VivaBench benchmark selected; skipping MIMIC-specific handler imports.")
else:
    raise ValueError(f"Unsupported BENCHMARK in config: {BENCHMARK}")



def handle_errors(func):
    """Generic error-handling decorator that catches and logs exceptions."""

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise

        return sync_wrapper


class ResourceHandler(ABC):
    """
    **Base resource handler** with FHIR-specific components removed.
    """

    @handle_errors
    async def fetch_result(self, patient_id: str, patient_hadm_id: str) -> Dict[str, Any]:
        """Fetch results asynchronously."""
        return await self._fetch_result_impl(patient_id, patient_hadm_id)

    @handle_errors
    async def generate_result_resource(
        self, result: Dict[str, Any], patient_id: str
    ) -> Dict[str, Any]:
        """Generate JSON results asynchronously."""
        return await self._generate_result_resource_impl(result, patient_id)

    @abstractmethod
    async def _fetch_result_impl(self, patient_id: str, patient_hadm_id: str) -> Dict[str, Any]:
        """Concrete implementation for fetching results."""
        ...

    @abstractmethod
    async def _generate_result_resource_impl(self, result: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """Concrete implementation for generating JSON results."""
        ...


class LabRequestHandler(ResourceHandler):
    
    def __init__(self, request: LabRequest, patient_data: pd.DataFrame):
        self.request = request
        self.patient_data = patient_data

    async def _fetch_result_impl(self, patient_id: str, patient_hadm_id: str) -> Dict[str, Any]:
        """Fetch laboratory test results."""
        return await asyncio.to_thread(
            fetch_lab_results,
            self.patient_data,
            patient_id,
            patient_hadm_id,
            self.request,
        )

    async def _generate_result_resource_impl(self, result: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """Convert the result to JSON."""
        return await asyncio.to_thread(
            generate_lab_observation_resource,
            self.request,
            result,
            patient_id,
        )


class UrineRequestHandler(ResourceHandler):

    def __init__(self, request: UrineRequest, patient_data: pd.DataFrame):
        self.request = request
        self.patient_data = patient_data

    async def _fetch_result_impl(self, patient_id: str, patient_hadm_id: str) -> Dict[str, Any]:
        """Fetch urine test results."""
        return await asyncio.to_thread(
            fetch_urine_results,
            self.patient_data,
            patient_id,
            patient_hadm_id,
            self.request,
        )

    async def _generate_result_resource_impl(self, result: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """Convert the result to JSON."""
        return await asyncio.to_thread(
            generate_urine_observation_resource,
            self.request,
            result,
            patient_id,
        )


class MedicationRequestHandler(ResourceHandler):
    """Handles the MedicationRequest instance and processes the request."""

    def __init__(self, request: MedicationRequest, patient_data: pd.DataFrame):
        self.request = request
        self.patient_data = patient_data

    async def _fetch_result_impl(
        self, patient_id: str, patient_hadm_id: str
    ) -> Optional[pd.Series]:
        """
        Specialist implementation to fetch medication results.
        Since there's no ground truth, we'll simulate the medication confirmation.
        """
        return await asyncio.to_thread(
            fetch_medication_results,
            self.patient_data,
            patient_id,
            patient_hadm_id,
            self.request,
        )

    async def _generate_result_resource_impl(
        self,
        result: Optional[pd.Series],
        patient_id: str,
    ) -> Dict[str, Any]:
        """
        Specialist implementation to generate Observation resource from MedicationRequest.
        """
        return await asyncio.to_thread(
            generate_medication_observation_resource,
            self.request,
            result,
            patient_id,
        )


class PhysicalExamRequestHandler(ResourceHandler):
    """Handles the PhysicalExamRequest instance and fetches the result."""

    def __init__(self, request: PhysicalExamRequest, patient_data: pd.DataFrame):
        self.request = request
        self.patient_data = patient_data

    async def _fetch_result_impl(
        self, patient_id: str, patient_hadm_id: str
    ) -> Optional[str]:
        """Fetch the PE data for the patient."""
        return await asyncio.to_thread(
            fetch_pe_results,
            self.patient_data,
            patient_id,
            patient_hadm_id,
            self.request
        )

    async def _generate_result_resource_impl(
        self,
        result: Optional[str],
        patient_id: str,
    ) -> Dict[str, Any]:
        """Generate the Observation resource containing the PE data."""
        return await asyncio.to_thread(
            generate_pe_observation_resource,
            self.request,
            result,
            patient_id,
        )
    

class MicrobiologyRequestHandler(ResourceHandler):
    """Handles the MicrobiologyRequestFHIR instance and processes the request."""

    def __init__(self, request: MicrobiologyRequest, patient_data: pd.DataFrame):
        self.request = request
        self.patient_data = patient_data

    async def _fetch_result_impl(
        self, patient_id: str, patient_hadm_id: str
    ) -> Optional[pd.Series]:
        """Fetch the microbiology data for the patient."""
        return await asyncio.to_thread(
            fetch_microbiology_results,
            self.patient_data,
            patient_id,
            patient_hadm_id,
            self.request,
        )

    async def _generate_result_resource_impl(
        self,
        result: Optional[pd.Series],
        patient_id: str,
    ) -> Dict[str, Any]:
        """
        Specialist implementation to generate Observation resources for microbiology test.

        Returns a list of Observations: [MicroTestObservation, MicroOrgObservation, MicroSuscObservation]
        """
        # Generate the MimicObservationMicroTest resource

        return await asyncio.to_thread(
            generate_microbiology_observations,
            self.request,
            result,
            patient_id,
        )


class RadiologyRequestHandler(ResourceHandler):
    """
    Handle radiology requests, fetch results, and return non-FHIR data.
    """

    def __init__(self, request: RadiologyRequest, patient_data: pd.DataFrame):
        self.request = request
        self.patient_data = patient_data

    async def _fetch_result_impl(
        self, patient_id: str, patient_hadm_id: str
    ) -> Optional[Union[pd.Series, dict, str]]:
        """
        Fetch radiology test results asynchronously.

        Args:
            patient_id (str): Patient identifier.
            patient_hadm_id (str): Hospital admission identifier.

        Returns:
            Optional[Union[pd.Series, dict, str]]: Radiology results, or None.
        """
        return await asyncio.to_thread(
            fetch_radiology_results,
            self.patient_data,
            patient_id,
            patient_hadm_id,
            self.request,
        )

    async def _generate_result_resource_impl(
        self,
        result: Union[pd.Series, dict, str, None],
        patient_id: str,
    ) -> Dict[str, Any]:
        """
        Generate non-FHIR radiology report data.

        Args:
            result (Union[pd.Series, dict, str, None]): Retrieved radiology data.
            patient_id (str): Patient identifier.

        Returns:
            Dict[str, Any]: Dictionary containing radiology data.
        """
        return await asyncio.to_thread(
            generate_radiology_report_resource,
            self.request,
            result,
            patient_id,
        )



class ProcedureSearchRequestHandler(ResourceHandler):

    def __init__(
        self,
        request: ProcedureSearch,
        patient_data: pd.DataFrame,
        collection: Qdrant_Collection,
    ):
        self.request = request
        self.patient_data = patient_data
        self.collection = collection

    async def _fetch_result_impl(
        self, patient_id: str, patient_hadm_id: str
    ) -> Optional[List[dict]]:
        # Fetch procedure options from the vector database
        return await asyncio.to_thread(
            fetch_procedure_search_results,
            self.collection,
            patient_id,
            patient_hadm_id,
            self.request,
        )

    async def _generate_result_resource_impl(
        self,
        result: Union[List[dict], None],
        patient_id: str,
    ) -> Optional[Union[pd.Series, dict, str]]:
        return await asyncio.to_thread(
            generate_procedure_search_resource,
            self.request,
            result,
            patient_id,
        )


class ProcedureRequestHandler(ResourceHandler):
    def __init__(
        self,
        request: ProcedureRequest,
        patient_data: pd.DataFrame,
        # collection: Qdrant_Collection,
    ):
        self.request = request
        self.patient_data = patient_data
        # self.collection = collection

    async def _fetch_result_impl(
        self, patient_id: str, patient_hadm_id: str
    ) -> Optional[List[dict]]:
        # Fetch procedure options from the vector database
        return await asyncio.to_thread(
            fetch_procedure_request_results,
            # self.collection,
            patient_id,
            patient_hadm_id,
            self.request,
        )

    async def _generate_result_resource_impl(
        self,
        result: Union[List[dict], None],
        patient_id: str,
    ) -> Optional[Union[pd.Series, dict, str]]:
        return await asyncio.to_thread(
            generate_procedure_resource,
            self.request,
            result,
            patient_id,
        )
