# models.py

from collections import deque
import numpy as np
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class AdPlatformAction(Action):
    allocations: list[float] = Field(..., description="Budget allocation per campaign")
    bids: list[float] = Field(..., description="Bid per campaign")


class AdPlatformObservation(Observation):
    """Observation returned to the agent at each step"""
    step: int = Field(default=0, description="Current timestep in the episode")
    remaining_budget: float = Field(default=0.0, description="Remaining budget for the episode")
    campaign_performance: list[float] = Field(default=list, description="Conversion rate per campaign")

    reward: float = Field(default=0.0)
    done: bool = Field(default=False)


class AdPlatformState(State):
    step_count: int = Field(default=0, description="Internal step counter")
    total_budget: float = Field(default=1000.0, description="Total budget")
    remaining_budget: float = Field(default=1000.0, description="Remaining budget")

    base_conversion_rates: list[float] = [0.05, 0.03, 0.02]
    conversion_rates: list[float] = []

    base_competitor_bids: list[float] = [0.5, 0.4, 0.3]
    competitor_bids: list[float] = []
    prev_agent_bids: list[float] = []

    total_conversions: float = Field(default=0.0, description="Total conversions over episode")
    total_spend: float = Field(default=0.0, description="Total budget spent")
    spend_history: list[float] = Field(default=list, description="Spend per step")
    reward_buffer: deque = Field(default_factory=lambda: deque(maxlen=1), description="Buffer for delayed reward")

    max_steps: int = Field(default=30, description="Max steps per episode")
    max_fraction_per_step: float = Field(default=0.3, description="Maximum fraction of budget that can be spent in one step")

    penalty_alpha: float = Field(default=0.05, description="Spend penalty coefficient")
    penalty_beta: float = Field(default=2.0, description="Exponent for spend penalty")

    seed: int = 42
    alpha: float = 0.3  # competitor responsiveness

    seasonal_amplitude: float = 0.1
    seasonal_period: int = 24

    market_events: dict = Field(default_factory=dict)

    # --- Constructor for realism mode and seed ---
    def __init__(self, realism_mode="fixed", seed=42, **data):
        """
        realism_mode: str
            "fixed"    → fully fixed, deterministic conversion rates
            "realistic" → deterministic slight perturbation per campaign
        seed: int → used for deterministic perturbation
        """
        super().__init__(**data)
        self.seed = seed
        self.realism_mode = realism_mode
        self.base_rates = [0.05, 0.03, 0.02]  # baseline conversion per $1
    

    # --- Method to set conversion rates based on realism_mode ---
    def set_conversion_rates(self):
        import numpy as np
        np.random.seed(self.seed)
        if self.realism_mode == "fixed":
            # Fully deterministic
            self.conversion_rates = [r * 1.0 for r in self.base_rates]
        elif self.realism_mode == "realistic":
            # Slight deterministic perturbation
            self.conversion_rates = [r * (1 + 0.01 * ((i + self.seed) % 3)) for i, r in enumerate(self.base_rates)]
        else:
            raise ValueError(f"Unknown realism_mode: {self.realism_mode}")