from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import CalcAction, CalcObservation, CalcState

class CalcEnv(EnvClient[CalcAction, CalcObservation, CalcState]):
    """
    Client for the OpenEnv_calc environment.
    """

    def _step_payload(self, action: CalcAction) -> Dict[str, Any]:
        """Convert an Action object to the JSON data expected by the env server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CalcObservation]:
        """Convert a JSON response from the env server to StepResult."""
        obs_data = payload.get("observation", {})
        obs = CalcObservation.model_validate(obs_data)
        
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CalcState:
        """Convert a JSON response from the state endpoint to a State object."""
        return CalcState.model_validate(payload)





# if __name__ == "__main__":
#     env = CalcEnv()
#     env.connect()
#     env.reset()
#     env.close()