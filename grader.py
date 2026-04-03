from models import BudgetState
import numpy as np
def compute_score(state: BudgetState) -> float:

    # --- Conversion score ---
    max_possible = state.total_budget * max(state.conversion_rates)
    conversion_score = state.total_conversions / (max_possible + 1e-8)

    # --- Budget utilization ---
    utilization = state.total_spend / (state.total_budget + 1e-8)
    utilization_score = 1 - abs(1 - utilization)

    # --- Smooth pacing ---
    if len(state.spend_history) > 1:
        spend_std = np.std(state.spend_history)
        spend_mean = np.mean(state.spend_history) + 1e-8
        smoothness_score = 1 - (spend_std / spend_mean)
        smoothness_score = max(0.0, smoothness_score)
    else:
        smoothness_score = 0.0

    # --- Final weighted score ---
    final_score = (
        0.7 * conversion_score +
        0.2 * utilization_score +
        0.1 * smoothness_score
    )

    return float(max(0.0, min(1.0, final_score)))