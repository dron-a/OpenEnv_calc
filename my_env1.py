from dataclasses import dataclass
from openenv import OpenEnv, StepResult

# 1. Define what the agent "sees"
@dataclass
class CalcObservation:
    current_value: int

# 2. Define what the agent "does"
@dataclass
class CalcAction:
    operation: str  # "add" or "sub"
    amount: int

class CalculatorEnv(OpenEnv):
    def __init__(self):
        super().__init__()
        self.value = 0

    def reset(self) -> CalcObservation:
        self.value = 0  # Reset state
        return CalcObservation(current_value=self.value)

    def step(self, action: CalcAction) -> StepResult:
        if action.operation == "add":
            self.value += action.amount
        elif action.operation == "sub":
            self.value -= action.amount
            
        obs = CalcObservation(current_value=self.value)
        # We give a reward if they get closer to 10
        reward = 1.0 if self.value == 10 else 0.0
        done = self.value >= 10
        
        return StepResult(observation=obs, reward=reward, done=done)