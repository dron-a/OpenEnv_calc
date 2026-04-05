# server/auction_environment.py

import numpy as np
from openenv.core.env_server import Environment
from models import AdPlatformAction, AdPlatformObservation, AdPlatformState


# ---------------- RESET Function ----------------
def reset(state: AdPlatformState) -> AdPlatformObservation:
    s = state
    s.step_count = 0
    s.remaining_budget = s.total_budget
    s.total_conversions = 0.0
    s.total_spend = 0.0
    s.reward_buffer.clear()

    s.prev_agent_bids = [0.0] * len(s.conversion_rates)
    s.competitor_bids = s.base_competitor_bids.copy()

    return AdPlatformObservation(
        step=s.step_count,
        remaining_budget=s.remaining_budget,
        campaign_performance=s.conversion_rates
    )

# ---------------- STEP ----------------
def step(state: AdPlatformState, action: AdPlatformAction) -> AdPlatformObservation:
    s = state

    allocations = action.allocations
    bids = action.bids

    assert len(bids) == len(s.conversion_rates), "Bid length mismatch"
    assert len(allocations) == len(s.conversion_rates), "Allocation length mismatch"

    # ----------------------------
    # Generate competitor bids (adaptive + seeded variation)
    # ----------------------------
    new_competitor_bids = []
    for i, base_bid in enumerate(s.base_competitor_bids):
        rng = np.random.default_rng(s.seed + s.step_count + i)
        # ±10% variation of base
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
    allocations = [max(0.0, min(a, pacing_limit)) for a in allocations]

    # ----------------------------
    # Budget spend
    # ----------------------------
    spend = sum(allocations)
    spend = min(spend, s.remaining_budget)

    # Store total available BEFORE spend (important for ratio)
    total_available = s.remaining_budget

    s.remaining_budget -= spend
    s.total_spend += spend

    # --- Spend ratio (for carryover penalty) ---
    if total_available > 0:
        spend_ratio = spend / (total_available + 1e-8)
    else:
        spend_ratio = 0.0

    # --- Conversions (smooth win probability) ---
    conversions = 0.0
    for a, bid, cb, cr in zip(allocations, bids, s.competitor_bids, s.conversion_rates):
        win_prob = 1 / (1 + np.exp(cb - bid))  # sigmoid
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
    # Spend penalty (legitimate)
    # ----------------------------
    frac = spend / (s.total_budget + 1e-8)
    penalty = s.penalty_alpha * (frac ** s.penalty_beta)

    # ----------------------------
    # Carryover penalty (early aggressive spending)
    # ----------------------------
    if s.step_count < s.max_steps - 1:
        time_factor = s.step_count / s.max_steps  # increases over time
        carryover_penalty = 0.2 * (spend_ratio ** 2) * (1 - time_factor)

    # ----------------------------
    # Total reward
    # ----------------------------
    reward = delayed_reward - penalty - illegal_penalty - carryover_penalty

    # ----------------------------
    # Step bookkeeping
    # ----------------------------
    s.prev_agent_bids = bids.copy()
    s.step_count += 1
    done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0

    return AdPlatformObservation(
        step=s.step_count,
        remaining_budget=s.remaining_budget,
        campaign_performance=s.conversion_rates,
        reward=reward,
        done=done
    )
