from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import BudgetAction, BudgetObservation, BudgetState

class BudgetEnvironment(EnvClient[BudgetAction, BudgetObservation, BudgetState]):
    """
    Client for the OpenEnv_calc environment.
    """

    def _step_payload(self, action: BudgetAction) -> Dict[str, Any]:
        """Convert an Action object to the JSON data expected by the env server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[BudgetObservation]:
        """Convert a JSON response from the env server to StepResult."""
        obs_data = payload.get("observation", {})
        obs = BudgetObservation.model_validate(obs_data)
        
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: Dict[str, Any]) -> BudgetState:
        """Convert a JSON response from the state endpoint to a State object."""
        return BudgetState.model_validate(payload)





# if __name__ == "__main__":
#     env = CalcEnv()
#     env.connect()
#     env.reset()
#     env.close()