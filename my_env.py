from dataclasses import dataclass
from openenv import AutoEnv, AutoAction, register  # Add register here

@dataclass
class CalcAction(AutoAction):
    operation: str 
    amount: int

@dataclass
class CalcObservation:
    current_value: int

class CalculatorEnv(AutoEnv):
    def __init__(self, **kwargs):  # Expert tip: Always allow kwargs in AutoEnv
        super().__init__(**kwargs)
        self.value = 0

    def reset(self) -> CalcObservation:
        self.value = 0
        return CalcObservation(current_value=self.value)

    def step(self, action: CalcAction):
        if action.operation == "add":
            self.value += action.amount
        elif action.operation == "sub":
            self.value -= action.amount
        return {
            "observation": CalcObservation(current_value=self.value),
            "reward": 1.0 if self.value == 10 else 0.0,
            "done": self.value >= 10,
            "info": {}
        }

# THE EXPERT MOVE: Register the class with a unique ID
register(
    id="MyCalculator-v1",
    entry_point=CalculatorEnv  # Point directly to the class
)