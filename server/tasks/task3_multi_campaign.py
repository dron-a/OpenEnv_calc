# server/multi_campaign_environment.py

import numpy as np
from openenv.core.env_server import Environment
from models import AdPlatformAction, AdPlatformObservation, AdPlatformState


# ---------------- RESET ----------------
def reset(state: AdPlatformState, profile: dict | None = None) -> AdPlatformObservation:
    s = state

    # Apply historical campaign profile (overrides defaults with real data)
    s.apply_profile(profile)

    s.step_count = 0
    s.remaining_budget = s.total_budget
    s.total_conversions = 0.0
    s.total_spend = 0.0
    s.obs_history.clear()
    s.reward_buffer.clear()

    s.prev_agent_bids = [0.0] * len(s.base_conversion_rates)
    s.competitor_bids = s.base_competitor_bids.copy()

    # Default market events (overridden if profile supplies its own)
    if not s.market_events:
        s.market_events = {
            10: [1.2, 1.0, 0.8],
            20: [0.9, 1.3, 1.0]
        }

    s.conversion_rates = s.base_conversion_rates.copy()

    return AdPlatformObservation(
        step=s.step_count,
        remaining_budget=s.remaining_budget,
        campaign_performance=s.conversion_rates,
        competitor_bids=s.competitor_bids,
        obs_history=[]
    )


# ---------------- STEP ----------------
def step(state: AdPlatformState, action: AdPlatformAction) -> AdPlatformObservation:
    s = state

    allocations = action.allocations
    bids = action.bids

    assert len(bids) == len(s.base_conversion_rates), "Bid length mismatch"
    assert len(allocations) == len(s.base_conversion_rates), "Allocation length mismatch"

    # ----------------------------
    # Update conversion rates (profile seasonal + market events + seeded noise)
    # ----------------------------
    seasonal = s.get_seasonal_multiplier()
    updated_rates = []
    for i, base in enumerate(s.base_conversion_rates):
        event_multiplier = s.market_events.get(
            s.step_count, [1.0] * len(s.base_conversion_rates)
        )[i]
        rng = np.random.default_rng(s.seed + s.step_count + i)
        noise = rng.uniform(-0.02, 0.02)
        updated_rates.append(base * seasonal * event_multiplier * (1 + noise))

    s.conversion_rates = updated_rates

    # ----------------------------
    # Generate competitor bids using historically-grounded volatility
    # ----------------------------
    s.competitor_bids = [
        s.sample_competitor_bid(i) for i in range(len(s.base_competitor_bids))
    ]

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
    for a in allocations:
        if a < 0:
            illegal_penalty += 0.5
        elif a > pacing_limit:
            illegal_penalty += 0.2

    # ----------------------------
    # Clamp allocations
    # ----------------------------
    allocations = [max(0.0, min(a, pacing_limit)) for a in allocations]

    # ----------------------------
    # Budget spend
    # ----------------------------
    spend = sum(allocations)
    spend = min(spend, s.remaining_budget)
    total_available = s.remaining_budget
    s.remaining_budget -= spend
    s.total_spend += spend

    spend_ratio = spend / (total_available + 1e-8) if total_available > 0 else 0.0

    # ----------------------------
    # Conversions (win probability via sigmoid)
    # ----------------------------
    conversions = 0.0
    for a, bid, cb, cr in zip(allocations, bids, s.competitor_bids, s.conversion_rates):
        win_prob = 1 / (1 + np.exp(cb - bid))
        conversions += a * cr * win_prob
    s.total_conversions += conversions

    # Record step into rolling history
    s.record_step(spend=spend, conversions=conversions, allocations=allocations, bids=bids)

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

    reward = delayed_reward - spend_penalty - illegal_penalty - carryover_penalty

    s.prev_agent_bids = bids.copy()
    s.step_count += 1
    done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0

    return AdPlatformObservation(
        step=s.step_count,
        remaining_budget=s.remaining_budget,
        campaign_performance=s.conversion_rates,
        competitor_bids=s.competitor_bids,
        obs_history=list(s.obs_history),
        reward=reward,
        done=done
    )
