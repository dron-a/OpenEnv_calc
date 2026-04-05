# models.py

from collections import deque
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class DynamicAction(Action):
    allocations: list[float]
    bids: list[float]


class DynamicObservation(Observation):
    step: int = 0
    remaining_budget: float = 0.0
    campaign_performance: list[float] = []
    reward: float = 0.0
    done: bool = False


class DynamicState(State):
    step_count: int = 0
    total_budget: float = 1000.0
    remaining_budget: float = 1000.0

    base_conversion_rates: list[float] = [0.05, 0.03, 0.02]
    conversion_rates: list[float] = []

    base_competitor_bids: list[float] = [0.5, 0.4, 0.3]
    competitor_bids: list[float] = []
    prev_agent_bids: list[float] = []

    reward_buffer: deque = Field(default_factory=lambda: deque(maxlen=1))

    total_conversions: float = 0.0
    total_spend: float = 0.0

    max_steps: int = 30
    max_fraction_per_step: float = 0.3

    penalty_alpha: float = 0.05
    penalty_beta: float = 2.0

    seed: int = 42
    alpha: float = 0.3  # competitor responsiveness

    seasonal_amplitude: float = 0.1
    seasonal_period: int = 24

    market_events: dict = Field(default_factory=dict)