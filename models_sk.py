# models.py
# from dataclasses import dataclass
from pydantic import Field
from openenv.core.env_server import Action, Observation, State

# @dataclass
class CalcAction(Action):
    """Your custom action."""
    # command: str
    # amount:  int
    command: str = Field(..., description="The calculator command (e.g., add, sub)")
    amount: int = Field(..., description="The numeric value to use")

# @dataclass
class CalcObservation(Observation):
    """Your custom observation."""
    current_value: int = Field(default=0, description="The value shown to the user")
    # current_value: int = 0
    # reward: float = Field(default=0.0, description="The reward value")
    # done: bool = Field(default=False, description="The value indicating if the task is done")

# @dataclass
class CalcState(State):
    """Custom state fields."""
    current_sum: int = Field(default=0, description="The internal state value")