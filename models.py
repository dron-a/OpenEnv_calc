# models.py
# from dataclasses import dataclass
from pydantic import Field
from openenv.core.env_server import Action, Observation, State

# @dataclass
class MultiPlatformEnvAction(Action):
    """Your custom action."""
    platform : str = Field(... , description="Platform chosen for allocation")
    allocation:float = Field(default=0 , description="Amount allocated for the chosen Platform")

# @dataclass
class MultiPlatformEnvObservation(Observation):
    """Your custom observation."""
    remaining_budget:float = Field(default=0, description="Budget remaining after previous allocation")
    remaining_days:int = Field(default=0, description="Days remaining after previous allocation")
    reward: float = Field(default=0.0, description="The reward value")
    done: bool = Field(default=False, description="The value indicating if the task is done")

# @dataclass
class MultiPlatformEnvState(State):
    """Custom state fields."""
    remaining_budget:float = Field(default=0, description="Budget remaining after previous allocation")
    remaining_days:int = Field(default=0, description="Days remaining after previous allocation")
    previous_day_leads:float =  Field(default=0, description="Days remaining after previous allocation")
    total_budget: float = Field(default = 1000)
    total_campaign_days: int = Field(default=30)
