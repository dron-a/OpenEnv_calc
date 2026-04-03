from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import AdSpendAction, AdObservation, AdState


class AdEnvClient(EnvClient[AdSpendAction, AdObservation, AdState]):
    """
    Client for the OpenEnv Ad Bidding environment.
    """

    def _step_payload(self, action: AdSpendAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AdObservation]:
        obs_data = payload.get("observation", {})
        obs = AdObservation.model_validate(obs_data)

        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AdState:
        return AdState.model_validate(payload)