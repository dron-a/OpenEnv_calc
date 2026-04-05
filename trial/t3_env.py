# server/multi_campaign_environment.py

import numpy as np
from openenv.core.env_server import Environment
from models import DynamicAction, DynamicObservation, DynamicState


class MultiCampaignEnvironment(Environment):
    state_type = DynamicState

    def __init__(self):
        super().__init__()
        self._state = DynamicState()

    # ---------------- RESET ----------------
    def reset(self) -> DynamicObservation:
        s = self._state

        s.step_count = 0
        s.remaining_budget = s.total_budget
        s.total_conversions = 0.0
        s.total_spend = 0.0
        s.reward_buffer.clear()

        s.prev_agent_bids = [0.0] * len(s.base_conversion_rates)
        s.competitor_bids = s.base_competitor_bids.copy()

        # Seeded market events
        s.market_events = {
            10: [1.2, 1.0, 0.8],
            20: [0.9, 1.3, 1.0]
        }

        # Initialize conversion rates
        s.conversion_rates = s.base_conversion_rates.copy()

        return DynamicObservation(
            step=s.step_count,
            remaining_budget=s.remaining_budget,
            campaign_performance=s.conversion_rates
        )

    # ---------------- STEP ----------------
    def step(self, action: DynamicAction) -> DynamicObservation:
        s = self._state

        allocations = action.allocations
        bids = action.bids

        assert len(bids) == len(s.base_conversion_rates), "Bid length mismatch"
        assert len(allocations) == len(s.base_conversion_rates), "Allocation length mismatch"

        # ----------------------------
        # UPDATE CONVERSION RATES
        # ----------------------------
        updated_rates = []
        for i, base in enumerate(s.base_conversion_rates):
            # Seasonality
            seasonal = 1 + s.seasonal_amplitude * np.sin(
                2 * np.pi * (s.step_count + i + s.seed) / s.seasonal_period
            )
            # Market events
            event_multiplier = s.market_events.get(s.step_count, [1.0]*len(s.base_conversion_rates))[i]
            # Seeded noise
            rng = np.random.default_rng(s.seed + s.step_count + i)
            noise = rng.uniform(-0.02, 0.02)

            updated_rates.append(base * seasonal * event_multiplier * (1 + noise))

        s.conversion_rates = updated_rates

        # ----------------------------
        # Generate competitor bids (adaptive + seeded variation)
        # ----------------------------
        new_competitor_bids = []
        for i, base_bid in enumerate(s.base_competitor_bids):
            rng = np.random.default_rng(s.seed + s.step_count + i)
            # ±10% variation
            variation = rng.uniform(-0.1, 0.1)
            noisy_base = base_bid * (1 + variation)
            # Adaptive response to previous agent bid
            prev_ab = s.prev_agent_bids[i]
            agent_effect = 1 + s.alpha * (prev_ab / (base_bid + 1e-8) - 1)
            cb = max(0.01, noisy_base * agent_effect)
            new_competitor_bids.append(cb)

        s.competitor_bids = new_competitor_bids

        # ----------------------------
        # Determine pacing limit
        # ----------------------------
        if s.step_count == s.max_steps - 1:
            pacing_limit = s.remaining_budget  # last step → allow all
        else:
            pacing_limit = s.max_fraction_per_step * s.remaining_budget

        # ----------------------------
        # Illegal allocation penalty
        # ----------------------------
        illegal_penalty = 0.0
        for i, a in enumerate(allocations):
            if a < 0:
                illegal_penalty += 0.5
            elif a > pacing_limit:
                illegal_penalty += 0.2  # only penalize overspending early

        # ----------------------------
        # Clamp allocations
        # ----------------------------
        allocation = [max(0.0, min(a, pacing_limit)) for a in allocations]

        # ----------------------------
        # Budget spend
        # ----------------------------
        spend = sum(allocation)
        spend = min(spend, s.remaining_budget)
        total_available = s.remaining_budget
        s.remaining_budget -= spend
        s.total_spend += spend

        spend_ratio = spend / (total_available + 1e-8) if total_available > 0 else 0.0

        # ----------------------------
        # Conversions (adaptive + market + seasonality + smooth win probability)
        # ----------------------------
        conversions = 0.0
        for a, bid, cb, cr in zip(allocation, bids, s.competitor_bids, s.conversion_rates):
            # Win probability via sigmoid
            win_prob = 1 / (1 + np.exp(cb - bid))
            conversions += a * cr * win_prob
        s.total_conversions += conversions

        # ----------------------------
        # Delayed reward
        # ----------------------------
        s.reward_buffer.append(conversions)
        if len(s.reward_buffer) == s.reward_buffer.maxlen:
            delayed_reward = s.reward_buffer.popleft()
        else:
            delayed_reward = 0.0

        # ----------------------------
        # Spend penalty
        # ----------------------------
        frac = spend / (s.total_budget + 1e-8)
        spend_penalty = s.penalty_alpha * (frac ** s.penalty_beta)

        # ----------------------------
        # Carryover penalty
        # ----------------------------
        carryover_penalty = 0.0
        if s.step_count < s.max_steps - 1:
            time_factor = s.step_count / s.max_steps
            carryover_penalty = 0.2 * (spend_ratio ** 2) * (1 - time_factor)

        # ----------------------------
        # Total reward
        # ----------------------------
        reward = delayed_reward - spend_penalty - illegal_penalty - carryover_penalty

        # ----------------------------
        # Step bookkeeping
        # ----------------------------
        s.prev_agent_bids = bids.copy()
        s.step_count += 1
        done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0

        return DynamicObservation(
            step=s.step_count,
            remaining_budget=s.remaining_budget,
            campaign_performance=s.conversion_rates,
            reward=reward,
            done=done
        )

    # ---------------- STATE ----------------
    @property
    def state(self):
        s = self._state
        return {
            "step_count": s.step_count,
            "remaining_budget": s.remaining_budget,
            "conversion_rates": s.conversion_rates,
            "competitor_bids": s.competitor_bids,
            "total_conversions": s.total_conversions,
            "total_spend": s.total_spend,
            "done": s.step_count >= s.max_steps or s.remaining_budget <= 0.0
        }

    # ---------------- MAX SCORE ----------------
    @property
    def max_possible_conversions(self):
        s = self._state
        return s.total_budget * max(s.base_conversion_rates)