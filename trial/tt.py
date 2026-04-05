# Determine pacing limit
if s.step_count == s.max_steps - 1:
    # Last step → allow spending all remaining budget
    pacing_limit = s.remaining_budget
else:
    # Normal pacing per step
    pacing_limit = s.max_fraction_per_step * s.remaining_budget

illegal_penalty = 0.0
for i, a in enumerate(action.allocation):
    if a < 0:
        # Negative allocation → strong illegal penalty
        illegal_penalty += 0.5
    elif a > pacing_limit:
        # Only penalize overspending if not last step
        illegal_penalty += 0.2

# Clamp allocation for actual spend
allocation = [max(0.0, min(a, pacing_limit)) for a in action.allocation]