from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import AdPlatformAction, AdPlatformObservation, AdPlatformState


class AdPlatformClient(EnvClient[AdPlatformAction, AdPlatformObservation, AdPlatformState]):
    """
    Client for the AdPlatform environment.
    """

    def _step_payload(self, action: AdPlatformAction) -> Dict[str, Any]:
        """Convert an Action object to the JSON data expected by the env server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AdPlatformObservation]:
        """Convert a JSON response from the env server to StepResult."""
        obs_data = payload.get("observation", {})
        obs = AdPlatformObservation.model_validate(obs_data)

        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AdPlatformState:
        """Convert a JSON response from the state endpoint to a State object."""
        return AdPlatformState.model_validate(payload)


