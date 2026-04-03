from pydantic import Field
from openenv.core.env_server import Action, Observation, State

class AdSpendAction(Action):
    """The action taken by the RL agent."""
    platform: str = Field(..., description="The platform to spend on (e.g., google, facebook, youtube)")
    spend_amount: float = Field(..., description="The amount of budget to spend today")

class AdObservation(Observation):
    """What the agent sees at each step."""
    remaining_budget: float = Field(default=1000.0, description="Budget left in dollars")
    days_remaining: int = Field(default=30, description="Days left in the campaign")
    previous_day_leads: int = Field(default=0, description="Leads generated in the previous step")

class AdState(State):
    """Internal state tracked by the environment server."""
    current_budget: float = Field(default=1000.0)
    days_passed: int = Field(default=0)
    previous_day_leads: int = Field(default=0)
    total_duration: int = Field(default=30)