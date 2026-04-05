# server/multi_task_environment.py

import numpy as np
from typing import Literal
from openenv.core.env_server import Environment
import server.tasks as tasks
from models import AdPlatformState


class AdPlatformEnvironment(Environment):
    """
    Unified environment supporting three tasks:
    1. Budget optimization
    2. Auction bidding
    3. Multi-campaign allocation
    """

    state_type = None  # dynamic depending on task

    def __init__(self, task: Literal["budget", "auction", "multi_campaign"] = "budget"):
        super().__init__()
        self.task = task
        self._state = AdPlatformState()
        self.state_type = AdPlatformState

        # self._state = None
        # if task == "budget":
        #     self._state = BudgetState()
        #     self.state_type = BudgetState
        # elif task == "auction":
        #     self._state = AuctionState()
        #     self.state_type = AuctionState
        # elif task == "multi_campaign":
        #     self._state = DynamicState()
        #     self.state_type = DynamicState
        # else:
        #     raise ValueError(f"Unknown task: {task}")

    # ---------------- RESET ----------------
    def reset(self, task: Literal["budget", "auction", "multi_campaign"] = None, realism_mode: str | None = None):
        """
        Reset the environment depending on the task
        """
        s = self._state
        if task is not None:
            self.task = task

        if self.task == "budget":
            return tasks.reset_budget(s, realism_mode)
        elif self.task == "auction":
            return tasks.reset_auction(s)
        elif self.task == "multi_campaign":
            return tasks.reset_multi_campaign(s)


    # ---------------- STEP ----------------
    def step(self, action):
        """
        Step the environment depending on the task
        """
        if self.task == "budget":
            print(self.task, "step")
            return tasks.step_budget(self._state, action)
        elif self.task == "auction":
            return tasks.step_auction(self._state, action)
        elif self.task == "multi_campaign":
            return tasks.step_multi_campaign(self._state, action)


    # ---------------- STATE ----------------
    @property
    def state(self):
        s = self._state
        if self.task == "budget":
            return {
            "step_count": s.step_count,
            "remaining_budget": s.remaining_budget,
            "campaign_performance": s.conversion_rates.copy(),
            "reward_buffer": list(s.reward_buffer),  # has to be JSON-serializable
            "total_conversions": s.total_conversions,
            "total_spend": s.total_spend,
            "done": s.step_count >= s.max_steps or s.remaining_budget <= 0.0,
        }
        else:
            return {
                "step_count": s.step_count,
                "remaining_budget": s.remaining_budget,
                "competitor_bids": getattr(s, "competitor_bids", []),
                "conversion_rates": getattr(s, "conversion_rates", []),
                "total_conversions": getattr(s, "total_conversions", 0.0),
                "total_spend": getattr(s, "total_spend", 0.0),
                "done": s.step_count >= s.max_steps or s.remaining_budget <= 0.0,
            }


    # ---------------- MAX SCORE ----------------
    @property
    def max_possible_conversions(self):
        s = self._state
        if self.task == "budget":
            return s.total_budget * max(getattr(s, "conversion_rates", [1.0]))
        else:
            return s.total_budget * max(getattr(s, "conversion_rates", [1.0]))