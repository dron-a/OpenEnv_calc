# models.py

from collections import deque
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class AuctionAction(Action):
    allocations: list[float] = Field(..., description="Budget allocation per campaign")
    bids: list[float] = Field(..., description="Bid per campaign")


class AuctionObservation(Observation):
    step: int = 0
    remaining_budget: float = 0.0
    campaign_performance: list[float] = []
    reward: float = 0.0
    done: bool = False


class AuctionState(State):
    step_count: int = 0
    remaining_budget: float = 1000.0
    total_budget: float = 1000.0

    conversion_rates: list[float] = [0.05, 0.03, 0.02]
    base_competitor_bids: list[float] = [0.5, 0.4, 0.3]

    competitor_bids: list[float] = []
    prev_agent_bids: list[float] = []

    total_conversions: float = 0.0
    total_spend: float = 0.0

    reward_buffer: deque = Field(default_factory=lambda: deque(maxlen=1))

    max_steps: int = 30
    max_fraction_per_step: float = 0.3

    penalty_alpha: float = 0.05
    penalty_beta: float = 2.0

    seed: int = 42
    alpha: float = 0.3  # competitor responsiveness