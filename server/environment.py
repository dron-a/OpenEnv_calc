# server/my_environment.py
import uuid

import numpy as np
from openenv.core.env_server import Environment
from models import MultiPlatformEnvAction, MultiPlatformEnvObservation, MultiPlatformEnvState

import inspect
print(f"MultiPlatformEnvState signature: {inspect.signature(MultiPlatformEnvState.__init__)}")

class MultiPlatformEnv(Environment):
    state_type = MultiPlatformEnvState
    def __init__(self):
        super().__init__()
        self.platforms = {
            'email': {'conv_rate': 0.2195, 'cpc': 37.18, 'max_daily_clicks': 270},
            'facebook': {'conv_rate': 0.2201, 'cpc': 37.35, 'max_daily_clicks': 268},
            'google': {'conv_rate': 0.2200, 'cpc': 37.33, 'max_daily_clicks': 266},
            'instagram': {'conv_rate': 0.2218, 'cpc': 37.21, 'max_daily_clicks': 268},
            'whatsapp': {'conv_rate': 0.2208, 'cpc': 37.36, 'max_daily_clicks': 268},
            'youtube': {'conv_rate': 0.2200, 'cpc': 37.07, 'max_daily_clicks': 270}
        }
        self.rng = np.random.default_rng()
        self._state = MultiPlatformEnvState(total_budget=1000, total_campaign_days=30)


    def reset(self) -> MultiPlatformEnvObservation:
        self._state = MultiPlatformEnvState(total_budget=1000, total_campaign_days=30)
        return MultiPlatformEnvObservation(
            remaining_budget=self._state.remaining_budget,
            remaining_days=self._state.remaining_days,
            reward=0.0,
            done=False
        )

    def step(self, action: MultiPlatformEnvAction) -> MultiPlatformEnvObservation:
        platform = action.platform
        allocation = action.allocation
        done = False
        if self._state.remaining_days <= 0 or self._state.remaining_budget < allocation :
            return MultiPlatformEnvObservation(
                remaining_budget=self._state.remaining_budget,
                remaining_days=self._state.remaining_days,
                reward=0.0,
                done=True
            )

        pre_budget = self._state.remaining_budget
        pre_days = self._state.remaining_days
        self._state.remaining_budget -= allocation
        self._state.remaining_days -= 1

        reward, metadata = self._calculate_effective_reward(platform, allocation, pre_budget, pre_days)

        return MultiPlatformEnvObservation(
            remaining_budget=self._state.remaining_budget,
            remaining_days=self._state.remaining_days,
            reward=reward,
            done=done)


    def _calculate_effective_reward(self, platform, allocation, remaining_budget, remaining_days):
        noise_scale = 0.15
        current_platform = self.platforms[platform]

        val = (allocation/current_platform['cpc']) * current_platform['conv_rate']
        leads  = float(np.maximum(0.0, self.rng.normal(val, val * noise_scale)))

        # Penalise spending more than your fair daily share
        fair_share = remaining_budget / remaining_days
        overspend = max(0.0, allocation - fair_share)
        reward = leads - 0.1 *overspend # 0.1 - raise it if the agent blows budget too fast, lower it if it under-spends.
        return reward,  {'leads': leads, 'fair_share': fair_share, 'overspend': overspend}



    @property
    def state(self) -> MultiPlatformEnvState:
        return self._state.model_dump()
