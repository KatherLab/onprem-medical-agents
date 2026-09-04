from ..configs.agent_config import BENCHMARK

if BENCHMARK == "VIVABENCH":
    from .tool_vivabench import (
        finish,
        request_blood_test,
        request_bedside_test,
        request_microbiology,
        request_other_investigation,
        request_physical_exam,
        request_radiology,
        request_urine_test,
        prescribe_medication,
        request_procedure,
    )

    TOOLS = [
        request_physical_exam,
        request_blood_test,
        request_urine_test,
        request_bedside_test,
        request_radiology,
        request_microbiology,
        request_other_investigation,
        # prescribe_medication,
        # request_procedure,
        finish,
    ]
else:
    from .tool_execs import *

    TOOLS = [
        request_physical_exam,
        request_blood_test, 
        request_urine_test,
        request_radiology,
        request_microbiology,
        prescribe_medication,
        search_procedure,
        request_procedure,
        finish,
             ]
