# server/budget_environment.py

import numpy as np
from openenv.core.env_server import Environment
from models import BudgetAction, BudgetObservation, BudgetState


class BudgetEnvironment(Environment):
    state_type = BudgetState

    def __init__(self):
        super().__init__()
        self._state = BudgetState()

    # ------------------------------
    # RESET
    # ------------------------------
    def reset(self) -> BudgetObservation:
        s = self._state

        # Reset core state
        s.step_count = 0
        s.remaining_budget = s.total_budget
        s.reward_buffer.clear()

        np.random.seed(s.seed)

        # Set conversion rates (fixed vs realistic)
        if s.realism_mode == "fixed":
            s.conversion_rates = s.base_rates.copy()

        elif s.realism_mode == "realistic":
            s.conversion_rates = [
                r * (1 + 0.01 * ((i + s.seed) % 3))
                for i, r in enumerate(s.base_rates)
            ]
        else:
            raise ValueError(f"Unknown realism_mode: {s.realism_mode}")

        return BudgetObservation(
            step=s.step_count,
            remaining_budget=s.remaining_budget,
            campaign_performance=s.conversion_rates
        )

    # ------------------------------
    # STEP
    # ------------------------------
    def step(self, action: BudgetAction) -> BudgetObservation:
        s = self._state

        # Validate action length
        assert len(action.allocation) == len(s.conversion_rates), "Allocation length mismatch"
        
        # Determine pacing limit
        if s.step_count == s.max_steps - 1:
            # Last step → allow spending all remaining budget
            pacing_limit = s.remaining_budget
        else:
            # Normal pacing per step
            pacing_limit = s.max_fraction_per_step * s.remaining_budget

         # --- Check for illegal actions ---
        illegal_penalty = 0.0

        for i, a in enumerate(action.allocation):
            if a < 0:
                # Negative allocation → strong illegal penalty
                illegal_penalty += 0.5
            elif a > pacing_limit:
                # Only penalize overspending if not last step
                # smaller penalty for trying to overspend
                illegal_penalty += 0.2

        # Clamp allocation to legal values for actual step calculation
        allocation = [max(0.0, min(a, pacing_limit)) for a in action.allocation]

        

        # --- Budget spend ---
        budget_spent = sum(allocation)

        # Prevent overspending remaining budget
        budget_spent = min(budget_spent, s.remaining_budget)
        s.remaining_budget -= budget_spent

        # --- Compute conversions ---
        conversions = sum(a * c for a, c in zip(allocation, s.conversion_rates))
        s.total_conversions += conversions
        s.total_spend += budget_spent
        s.spend_history.append(budget_spent)

        # --- Delayed reward ---
        s.reward_buffer.append(conversions)
        if len(s.reward_buffer) == s.reward_buffer.maxlen:
            delayed_reward = s.reward_buffer.popleft()
        else:
            delayed_reward = 0.0

        # --- Spend penalty (legitimate) ---
        fraction_spent = budget_spent / (s.total_budget + 1e-8)
        penalty = s.penalty_alpha * (fraction_spent ** s.penalty_beta)

        # --- Total reward (include illegal action penalty) ---
        reward = delayed_reward - penalty - illegal_penalty

        # --- Step bookkeeping ---
        s.step_count += 1

        done = (
            s.step_count >= s.max_steps
            or s.remaining_budget <= 0.0
        )

        return BudgetObservation(
            step=s.step_count,
            remaining_budget=s.remaining_budget,
            campaign_performance=s.conversion_rates,
            reward=reward,
            done=done
        )

    # ------------------------------
    # STATE (for API)
    # ------------------------------
    @property
    def state(self) -> BudgetState:
        return self._state.model_dump()

    # ------------------------------
    # MAX SCORE (for grading)
    # ------------------------------
    @property
    def max_possible_conversions(self) -> float:
        s = self._state
        return s.total_budget * max(s.conversion_rates)