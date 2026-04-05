# server/multi_task_environment.py

import numpy as np
from typing import Literal
from openenv.core.env_server import Environment
from models import (
    BudgetAction,
    BudgetObservation,
    BudgetState,
    AuctionAction,
    AuctionObservation,
    AuctionState,
    DynamicAction,
    DynamicObservation,
    DynamicState,
)


class MultiTaskEnvironment(Environment):
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
        self._state = None
        if task == "budget":
            self._state = BudgetState()
            self.state_type = BudgetState
        elif task == "auction":
            self._state = AuctionState()
            self.state_type = AuctionState
        elif task == "multi_campaign":
            self._state = DynamicState()
            self.state_type = DynamicState
        else:
            raise ValueError(f"Unknown task: {task}")

    # ---------------- RESET ----------------
    def reset(self, realism_mode: str | None = None):
        """
        Reset the environment depending on the task
        """
        s = self._state
        if self.task == "budget":
            s.step_count = 0
            s.remaining_budget = s.total_budget
            s.reward_buffer.clear()
            if realism_mode is not None:
                s.realism_mode = realism_mode

            if getattr(s, "realism_mode", "fixed") == "fixed":
                s.conversion_rates = s.base_rates.copy()
            else:
                s.conversion_rates = [
                    r * (1 + 0.01 * ((i + s.seed) % 3)) for i, r in enumerate(s.base_rates)
                ]

            return BudgetObservation(
                step=s.step_count,
                remaining_budget=s.remaining_budget,
                campaign_performance=s.conversion_rates,
            )

        elif self.task == "auction":
            s.step_count = 0
            s.remaining_budget = s.total_budget
            s.total_conversions = 0.0
            s.total_spend = 0.0
            s.reward_buffer.clear()
            s.prev_agent_bids = [0.0] * len(s.conversion_rates)
            s.competitor_bids = s.base_competitor_bids.copy()

            return AuctionObservation(
                step=s.step_count,
                remaining_budget=s.remaining_budget,
                campaign_performance=s.conversion_rates,
            )

        elif self.task == "multi_campaign":
            s.step_count = 0
            s.remaining_budget = s.total_budget
            s.total_conversions = 0.0
            s.total_spend = 0.0
            s.reward_buffer.clear()
            s.prev_agent_bids = [0.0] * len(s.base_conversion_rates)
            s.competitor_bids = s.base_competitor_bids.copy()
            s.market_events = {10: [1.2, 1.0, 0.8], 20: [0.9, 1.3, 1.0]}
            s.conversion_rates = s.base_conversion_rates.copy()

            return DynamicObservation(
                step=s.step_count,
                remaining_budget=s.remaining_budget,
                campaign_performance=s.conversion_rates,
            )

    # ---------------- STEP ----------------
    def step(self, action):
        """
        Step the environment depending on the task
        """
        if self.task == "budget":
            return self._step_budget(action)
        elif self.task == "auction":
            return self._step_auction(action)
        elif self.task == "multi_campaign":
            return self._step_multi_campaign(action)

    # ---------------- BUDGET STEP ----------------
    def _step_budget(self, action: BudgetAction):
        s: BudgetState = self._state

        allocation = [max(0.0, min(a, s.max_fraction_per_step * s.remaining_budget)) for a in action.allocation]
        budget_spent = min(sum(allocation), s.remaining_budget)
        s.remaining_budget -= budget_spent

        conversions = sum(a * c for a, c in zip(allocation, s.conversion_rates))
        s.total_conversions += conversions
        s.total_spend += budget_spent

        s.reward_buffer.append(conversions)
        delayed_reward = s.reward_buffer.popleft() if len(s.reward_buffer) == s.reward_buffer.maxlen else 0.0

        penalty = s.penalty_alpha * ((budget_spent / (s.total_budget + 1e-8)) ** s.penalty_beta)
        reward = delayed_reward - penalty

        s.step_count += 1
        done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0

        return BudgetObservation(
            step=s.step_count,
            remaining_budget=s.remaining_budget,
            campaign_performance=s.conversion_rates,
            reward=reward,
            done=done,
        )

    # ---------------- AUCTION STEP ----------------
    def _step_auction(self, action: AuctionAction):
        s: AuctionState = self._state

        allocations, bids = action.allocations, action.bids
        pacing_limit = s.remaining_budget if s.step_count == s.max_steps - 1 else s.max_fraction_per_step * s.remaining_budget

        illegal_penalty = 0.0
        for a in allocations:
            if a < 0:
                illegal_penalty += 0.5
            elif a > pacing_limit:
                illegal_penalty += 0.2

        allocation = [max(0.0, min(a, pacing_limit)) for a in allocations]
        spend = min(sum(allocation), s.remaining_budget)
        s.remaining_budget -= spend
        s.total_spend += spend

        # Update competitor bids
        s.competitor_bids = [
            max(0.01, b * (1 + 0.1 * ((i + s.seed) % 2)) * (1 + s.alpha * (pa / (b + 1e-8) - 1)))
            for i, (b, pa) in enumerate(zip(s.base_competitor_bids, s.prev_agent_bids))
        ]

        conversions = sum(
            a * cr / (1 + np.exp(cb - bid))
            for a, bid, cb, cr in zip(allocation, bids, s.competitor_bids, s.conversion_rates)
        )
        s.total_conversions += conversions

        s.reward_buffer.append(conversions)
        delayed_reward = s.reward_buffer.popleft() if len(s.reward_buffer) == s.reward_buffer.maxlen else 0.0

        penalty = s.penalty_alpha * ((spend / (s.total_budget + 1e-8)) ** s.penalty_beta)
        reward = delayed_reward - penalty - illegal_penalty

        s.prev_agent_bids = bids.copy()
        s.step_count += 1
        done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0

        return AuctionObservation(
            step=s.step_count,
            remaining_budget=s.remaining_budget,
            campaign_performance=s.conversion_rates,
            reward=reward,
            done=done,
        )

    # ---------------- MULTI-CAMPAIGN STEP ----------------
    def _step_multi_campaign(self, action: DynamicAction):
        s: DynamicState = self._state
        allocations, bids = action.allocations, action.bids

        # Update conversion rates with seasonality and market events
        s.conversion_rates = [
            base * (1 + s.seasonal_amplitude * np.sin(2 * np.pi * (s.step_count + i + s.seed) / s.seasonal_period))
            * s.market_events.get(s.step_count, [1.0] * len(s.base_conversion_rates))[i]
            for i, base in enumerate(s.base_conversion_rates)
        ]

        # Competitor bids
        s.competitor_bids = [
            max(0.01, b * (1 + 0.1 * ((i + s.seed) % 2)) * (1 + s.alpha * (pa / (b + 1e-8) - 1)))
            for i, (b, pa) in enumerate(zip(s.base_competitor_bids, s.prev_agent_bids))
        ]

        pacing_limit = s.remaining_budget if s.step_count == s.max_steps - 1 else s.max_fraction_per_step * s.remaining_budget
        illegal_penalty = sum(0.5 if a < 0 else 0.2 if a > pacing_limit else 0.0 for a in allocations)
        allocation = [max(0.0, min(a, pacing_limit)) for a in allocations]

        spend = min(sum(allocation), s.remaining_budget)
        total_available = s.remaining_budget
        s.remaining_budget -= spend
        s.total_spend += spend
        spend_ratio = spend / (total_available + 1e-8) if total_available > 0 else 0.0

        conversions = sum(
            a * cr / (1 + np.exp(cb - bid))
            for a, bid, cb, cr in zip(allocation, bids, s.competitor_bids, s.conversion_rates)
        )
        s.total_conversions += conversions

        s.reward_buffer.append(conversions)
        delayed_reward = s.reward_buffer.popleft() if len(s.reward_buffer) == s.reward_buffer.maxlen else 0.0

        penalty = s.penalty_alpha * ((spend / (s.total_budget + 1e-8)) ** s.penalty_beta)
        carryover_penalty = 0.2 * (spend_ratio ** 2) * (1 - s.step_count / s.max_steps) if s.step_count < s.max_steps - 1 else 0.0

        reward = delayed_reward - penalty - carryover_penalty - illegal_penalty

        s.prev_agent_bids = bids.copy()
        s.step_count += 1
        done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0

        return DynamicObservation(
            step=s.step_count,
            remaining_budget=s.remaining_budget,
            campaign_performance=s.conversion_rates,
            reward=reward,
            done=done,
        )

    # ---------------- STATE ----------------
    @property
    def state(self):
        s = self._state
        if self.task == "budget":
            return {
                "step_count": s.step_count,
                "remaining_budget": s.remaining_budget,
                "conversion_rates": getattr(s, "conversion_rates", []),
                "total_conversions": getattr(s, "total_conversions", 0.0),
                "total_spend": getattr(s, "total_spend", 0.0),
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