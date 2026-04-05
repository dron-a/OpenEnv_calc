# server/my_environment.py
import uuid
from openenv.core.env_server import Environment
from models import CalcAction, CalcObservation, CalcState

import inspect
print(f"CalcState signature: {inspect.signature(CalcState.__init__)}")

class CalcEnvironment(Environment):
    state_type = CalcState
    def __init__(self):
        super().__init__()
        self._state = CalcState()
        # Debug print to verify it's there inside the container
        print(f"DEBUG: Initialized state value: {self._state.current_sum}")

    def reset(self) -> CalcObservation:
        self._state = CalcState()
        return CalcObservation()

    def step(self, action: CalcAction) -> CalcObservation:
        # Implement your logic here
        self._state.step_count += 1
        outcome = self._execute_command(action.command, action.amount)
        return CalcObservation(current_value=outcome["current_value"],reward=outcome["reward"],done=outcome["done"])
    
    def _execute_command(self, command: str, amount: int) -> dict:
        # Your custom logic
        if command == "add":
            self._state.current_sum += amount
            return {"current_value": self._state.current_sum, "reward": 1.0 if self._state.current_sum == 10 else 0.0, "done": self._state.current_sum >= 10}
        elif command == "sub":
            self._state.current_sum -= amount
            return {"current_value": self._state.current_sum, "reward": 1.0 if self._state.current_sum == 10 else 0.0, "done": self._state.current_sum >= 10}
        else:
            return {"current_value": self._state.current_sum, "reward": 0.0, "done": False}

    @property
    def state(self) -> CalcState:
        print(f"DEBUG: API is requesting state. Current value: {self._state.current_sum}")
        # return self._state.model_dump()
        # We use model_dump() to turn the CalcState into a plain dictionary
        data = self._state.model_dump()
        print(f"DEBUG: Sending dictionary to API: {data}")
        return data