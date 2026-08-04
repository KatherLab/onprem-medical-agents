import os
import json
from typing import List, Literal
from pydantic import BaseModel
from agents import Agent, handoff, RunContextWrapper, ModelSettings, HandoffInputData
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.extensions import handoff_filters

from ..prompts import CRITIC_SYSTEM_PROMPT
from ..configs.agent_config import CRITIC_MODEL


class DiagnosisHypothesis(BaseModel):
    diagnosis: str
    probability: Literal["High", "Medium", "Low"]

class DifferentialDiagnosisData(BaseModel):
    patient_summary: str
    differential_diagnosis: List[DiagnosisHypothesis]


critic_agent = Agent(
    name="DiagnosticCriticAgent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
{CRITIC_SYSTEM_PROMPT}""",
    model=CRITIC_MODEL,
    model_settings=ModelSettings(
        temperature=0.05,
        max_tokens=8192,
    )
)

# Handoff callback (optional, but useful for logging).
async def on_critic_handoff(
    ctx: RunContextWrapper[None], 
    input_data: DifferentialDiagnosisData
):
    """This function is called immediately when the handoff is triggered."""
    # ctx must remain in the signature even when it is not used here.
    print("="*20)
    print("HANDOFF TRIGGERED: Doctor Agent -> Critic Agent")
    print(f"Critiquing DDx: {[d.diagnosis for d in input_data.differential_diagnosis]}")
    print("="*20)

# Create the handoff object.
critic_handoff = handoff(
    agent=critic_agent,
    tool_name_override="request_diagnostic_critique",  # Tool name.
    tool_description_override=(
        "Requests a safety review of your current differential diagnosis from a specialist critic. "
        "Use this tool after you have formed a preliminary diagnostic plan but before ordering definitive or high-cost tests. "
        "Provide a summary of the case and your list of hypotheses."
    ),
    input_type=DifferentialDiagnosisData,  # Required handoff input schema.
    on_handoff=on_critic_handoff,
)




