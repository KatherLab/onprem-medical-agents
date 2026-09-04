import os
from typing import List
from pydantic import BaseModel, Field
from agents import Agent, handoff, RunContextWrapper, ModelSettings

from ..prompts import EXECUTOR_SYSTEM_PROMPT
from ..configs.agent_config import EXECUTOR_MODEL
from ..tools import TOOLS


# --- 1. Define the handoff data structure. ---
# This is the data format that PlannerAgent (Med-Gemma) must provide.
class ExecutionPlan(BaseModel):
    """A structured plan of tasks for the Executor Agent to perform."""
    rationale: str = Field(..., description="The reasoning behind why these tasks are necessary.")
    task_list: List[str] = Field(..., description="A list of tasks described in natural language for the executor to complete.")

# --- 2. Create the Executor Agent. ---
# This agent has all action-oriented tools.
executor_agent = Agent(
    name="ExecutorAgent",
    instructions=EXECUTOR_SYSTEM_PROMPT,
    model=EXECUTOR_MODEL,
    # Its toolset contains all medical action tools.
    tools=TOOLS,
    model_settings=ModelSettings(
        # Require tool calls while allowing the model to return errors when necessary.
        tool_choice="required", 
        temperature=0.01,  # Use a low temperature for precision.
    )
)

# Handoff callback (optional, but useful for logging).
async def on_executor_handoff(
    ctx: RunContextWrapper[None], 
    input_data: ExecutionPlan
):
    """This function is called immediately when the handoff is triggered."""
    # ctx must remain in the signature even when it is not used here.
    print("="*20)
    print("HANDOFF TRIGGERED: Doctor Agent -> Executor Agent")
    print(f"TO-DO List: {[d.task for d in input_data.task_list]}")
    print("="*20)

# --- 3. Create the handoff from Planner to Executor. ---
# This handoff object is used by PlannerAgent (Med-Gemma).
executor_handoff = handoff(
    # Target the executor agent.
    agent=executor_agent,
    
    # This tool is called by PlannerAgent.
    tool_name_override="delegate_tasks_to_executor",
    
    tool_description_override=(
        "Delegates a list of clinical tasks to a specialized executor agent. "
        "Use this tool after you have formed a plan. Provide your reasoning and a clear list of tasks."
    ),
    
    # PlannerAgent must provide this Pydantic model as input.
    input_type=ExecutionPlan,
    on_handoff=on_executor_handoff,
)