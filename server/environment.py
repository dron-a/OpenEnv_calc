import random
import numpy as np
from openenv.core.env_server import Environment
from models import AdSpendAction, AdObservation, AdState


class AdBiddingEnv(Environment):
    state_type = AdState

    def __init__(self):
        super().__init__()
        self._state = AdState()

        # Real-world data extracted from nykaa_campaign_data.csv
        self.platforms = {
            'email': {'conv_rate': 0.2195, 'cpc': 37.18, 'max_daily_clicks': 270},
            'facebook': {'conv_rate': 0.2201, 'cpc': 37.35, 'max_daily_clicks': 268},
            'google': {'conv_rate': 0.2200, 'cpc': 37.33, 'max_daily_clicks': 266},
            'instagram': {'conv_rate': 0.2218, 'cpc': 37.21, 'max_daily_clicks': 268},
            'whatsapp': {'conv_rate': 0.2208, 'cpc': 37.36, 'max_daily_clicks': 268},
            'youtube': {'conv_rate': 0.2200, 'cpc': 37.07, 'max_daily_clicks': 270}
        }

    def reset(self) -> AdObservation:
        self._state = AdState(
            current_budget=1000.0,
            days_passed=0,
            previous_day_leads=0,
            total_duration=30
        )
        return self._get_obs()

    def _get_obs(self) -> AdObservation:
        return AdObservation(
            remaining_budget=self._state.current_budget,
            days_remaining=self._state.total_duration - self._state.days_passed,
            previous_day_leads=self._state.previous_day_leads
        )

    def step(self, action: AdSpendAction) -> AdObservation:
        self._state.days_passed += 1

        actual_spend = max(0.0, min(action.spend_amount, self._state.current_budget))

        # Fetch metrics for the specific platform
        platform_name = action.platform.lower()
        platform_data = self.platforms.get(platform_name, self.platforms['google'])

        cpc = platform_data['cpc']
        conv_rate = platform_data['conv_rate']
        base_max_clicks = platform_data['max_daily_clicks']

        # Add realistic market volatility (+/- 10% daily click availability)
        noise = random.uniform(0.9, 1.1)
        max_daily_clicks = int(base_max_clicks * noise)

        # Calculate Clicks
        affordable_clicks = int(actual_spend / cpc)
        actual_clicks = min(affordable_clicks, max_daily_clicks)

        # Generate leads probabilistically
        leads_generated = int(np.random.binomial(actual_clicks, conv_rate))

        # Update state
        self._state.current_budget -= actual_spend
        self._state.previous_day_leads = leads_generated

        # Check termination
        is_done = self._state.days_passed >= self._state.total_duration or self._state.current_budget <= 0.0

        return AdObservation(
            remaining_budget=self._state.current_budget,
            days_remaining=self._state.total_duration - self._state.days_passed,
            previous_day_leads=self._state.previous_day_leads,
            reward=float(leads_generated),
            done=is_done
        )

    @property
    def state(self) -> AdState:
        return self._state.model_dump()