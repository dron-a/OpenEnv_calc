from my_env import CalculatorEnv, CalcAction
from openenv import AutoEnv

# EXPERT BYPASS: Use from_cls to bypass the need for string registration.
# This works in 99% of OpenEnv 2026 versions for local development.
try:
    env = AutoEnv.from_cls(CalculatorEnv)
    print("--- Environment Initialized via from_cls ---")
except AttributeError:
    # If that fails, the SDK might just want the class in the main factory
    env = AutoEnv(CalculatorEnv)
    print("--- Environment Initialized via Direct Constructor ---")

obs = env.reset()
print(f"Start State: {obs}")

action = CalcAction(operation="add", amount=5)
result = env.step(action)
print(f"Result: {result}")