# server/multi_campaign_environment.py

import numpy as np
from openenv.core.env_server import Environment
from models import MultiCampaignAction, MultiCampaignObservation, MultiCampaignState
from collections import deque

class MultiCampaignEnvironment(Environment):
    state_type = MultiCampaignState

    def __init__(self):
        super().__init__()
        self._state = MultiCampaignState()

    # ---------------- RESET ----------------
    def reset(self) -> MultiCampaignObservation:
        s = self._state
        s.step_count = 0
        s.remaining_budget = s.total_budget
        s.total_conversions = 0.0
        s.total_spend = 0.0
        s.reward_buffer = deque(maxlen=3)

        s.prev_agent_bids = [0.0] * len(s.conversion_rates)
        s.competitor_bids = s.base_competitor_bids.copy()

        # Reset market dynamics
        s.market_multiplier = 1.0
        s.seasonality_phase = 0.0  # start of seasonality cycle

        return MultiCampaignObservation(
            step=s.step_count,
            remaining_budget=s.remaining_budget,
            campaign_performance=s.conversion_rates,
            market_multiplier=s.market_multiplier
        )

    # ---------------- STEP ----------------
    def step(self, action: MultiCampaignAction) -> MultiCampaignObservation:
        s = self._state

        allocations = action.allocations
        bids = action.bids

        assert len(bids) == len(s.conversion_rates), "Bid length mismatch"
        assert len(allocations) == len(s.conversion_rates), "Allocation length mismatch"

        # ----------------------------
        # Update market dynamics (seeded)
        # ----------------------------
        rng_market = np.random.default_rng(s.seed + s.step_count)
        market_shock = rng_market.normal(1.0, 0.05)  # ±5% market multiplier
        s.market_multiplier = max(0.8, min(1.2, s.market_multiplier * market_shock))

        # Seasonality factor (sinusoidal pattern)
        seasonality = 0.1 * np.sin(2 * np.pi * s.seasonality_phase)
        s.seasonality_phase += 1.0 / s.max_steps  # advance phase
        s.seasonality_phase %= 1.0

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
            pacing_limit = s.remaining_budget
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
                illegal_penalty += 0.2

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
            # Conversion is affected by market, seasonality, and stochastic seeded noise
            rng_conv = np.random.default_rng(s.seed + s.step_count + int(a*100))
            noise = rng_conv.normal(1.0, 0.02)  # ±2% deterministic noise
            conversions += a * cr * win_prob * s.market_multiplier * (1 + seasonality) * noise

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
        penalty = s.penalty_alpha * (frac ** s.penalty_beta)

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

        return MultiCampaignObservation(
            step=s.step_count,
            remaining_budget=s.remaining_budget,
            campaign_performance=s.conversion_rates,
            market_multiplier=s.market_multiplier,
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
            "competitor_bids": s.competitor_bids,
            "total_conversions": s.total_conversions,
            "total_spend": s.total_spend,
            "market_multiplier": s.market_multiplier,
            "done": s.step_count >= s.max_steps or s.remaining_budget <= 0.0
        }

    # ---------------- MAX SCORE ----------------
    @property
    def max_possible_conversions(self):
        s = self._state
        return s.total_budget * max(s.conversion_rates) * 1.2  # max market boost