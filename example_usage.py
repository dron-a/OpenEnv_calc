import asyncio
import random

from client import BudgetEnvironment
from models import BudgetAction

async def main():
    # Connect to the running OpenEnv server.
    base_url = "ws://localhost:8000"

    
    print(f"Connecting to environment at {base_url}...")

    async with BudgetEnvironment(base_url=base_url) as env:
        print("Successfully connected !")

        # 1. Reset the environment to start a new episode
        print("\n--- Resetting Environment ---")
        reset_result = await env.reset()
        print(f"Initial Observation:{reset_result.observation}")
        print(f"Is Done?: {reset_result.done}")

        done = False

        while not done:
            amounts = []
            #  Take a specific action
            amounts = [random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)]
            action1 = BudgetAction( allocation=amounts)
            step_result = await env.step(action1)

            # Inspect the Observation returned by that step
            print(f"Current Observation: {step_result.observation}")
            print(f"Reward received: {step_result.reward}")
            print(f"Is Done?: {step_result.done}")

            # 3. Inspect internal environment State (if necessary for debugging)
            print("\n--- Checking server-side State ---")
            state = await env.state()

            print(f"Current state: {state.remaining_budget}/{state.total_budget} , {state.step_count}/{state.max_steps} ")
            print(f"Total steps taken globally: {state.step_count}")

            done = step_result.done

        if step_result.done:
            print("\nEpisode finished successfully!")

if __name__ == "__main__":
    asyncio.run(main())
