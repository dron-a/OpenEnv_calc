# server/budget_environment.py

import numpy as np
from openenv.core.env_server import Environment
from models import AdPlatformAction, AdPlatformObservation, AdPlatformState


# ---------------- RESET ----------------
def reset(state: AdPlatformState, realism_mode: str | None = None,
          profile: dict | None = None) -> AdPlatformObservation:
    s = state

    # Apply historical campaign profile first (overrides defaults with real data)
    s.apply_profile(profile)

    if realism_mode is not None:
        s.realism_mode = realism_mode

    # Reset all episode state
    s.step_count = 0
    s.remaining_budget = s.total_budget
    s.total_conversions = 0.0
    s.total_spend = 0.0
    s.spend_history.clear()
    s.obs_history.clear()
    s.reward_buffer.clear()

    np.random.seed(s.seed)

    if s.realism_mode == "fixed":
        s.conversion_rates = s.base_conversion_rates.copy()
    elif s.realism_mode == "realistic":
        s.conversion_rates = [
            r * (1 + 0.01 * ((i + s.seed) % 3))
            for i, r in enumerate(s.base_conversion_rates)
        ]
    else:
        raise ValueError(f"Unknown realism_mode: {s.realism_mode}")

    return AdPlatformObservation(
        step=s.step_count,
        remaining_budget=s.remaining_budget,
        campaign_performance=s.conversion_rates,
        obs_history=[]
    )


# ---------------- STEP ----------------
def step(state: AdPlatformState, action: AdPlatformAction) -> AdPlatformObservation:
    s = state

    s.set_conversion_rates()

    assert len(action.allocations) == len(s.conversion_rates), "allocations length mismatch"

    # Determine pacing limit
    if s.step_count == s.max_steps - 1:
        pacing_limit = s.remaining_budget
    else:
        pacing_limit = s.max_fraction_per_step * s.remaining_budget

    # Check for illegal actions
    illegal_penalty = 0.0
    for a in action.allocations:
        if a < 0:
            illegal_penalty += 0.5
        elif a > pacing_limit:
            illegal_penalty += 0.2

    # Clamp allocations
    allocations = [max(0.0, min(a, pacing_limit)) for a in action.allocations]

    # Budget spend
    budget_spent = min(sum(allocations), s.remaining_budget)
    s.remaining_budget -= budget_spent

    # Compute conversions (seasonal multiplier applied if profile supplies one)
    seasonal = s.get_seasonal_multiplier()
    conversions = sum(a * c * seasonal for a, c in zip(allocations, s.conversion_rates))
    s.total_conversions += conversions
    s.total_spend += budget_spent
    s.spend_history.append(budget_spent)

    # Record step into rolling history
    s.record_step(spend=budget_spent, conversions=conversions, allocations=allocations, bids=[])

    # Delayed reward
    s.reward_buffer.append(conversions)
    if len(s.reward_buffer) == s.reward_buffer.maxlen:
        delayed_reward = s.reward_buffer.popleft()
    else:
        delayed_reward = 0.0

    # Spend penalty
    fraction_spent = budget_spent / (s.total_budget + 1e-8)
    penalty = s.penalty_alpha * (fraction_spent ** s.penalty_beta)

    reward = delayed_reward - penalty - illegal_penalty

    s.step_count += 1
    done = s.step_count >= s.max_steps or s.remaining_budget <= 0.0

    return AdPlatformObservation(
        step=s.step_count,
        remaining_budget=s.remaining_budget,
        campaign_performance=s.conversion_rates,
        obs_history=list(s.obs_history),
        reward=reward,
        done=done
    )
