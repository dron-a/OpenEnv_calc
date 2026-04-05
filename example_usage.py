import asyncio
import random

from client import AdPlatformClient
from models import AdPlatformAction

async def main():
    # Connect to the running OpenEnv server.
    base_url = "ws://localhost:8000"

    print(f"Connecting to environment at {base_url}...")

    async with AdPlatformClient(base_url=base_url) as env:
        print("Successfully connected !")

        # 1. Reset the environment to start a new episode
        print("\n--- Resetting Environment ---")
        reset_result = await env.reset()
        print(f"Initial Observation: {reset_result.observation}")
        print(f"Is Done?: {reset_result.done}")

        done = False

        while not done:
            # Random allocations and bids for 3 campaigns
            allocations = [random.uniform(1, 100) for _ in range(3)]
            bids = [random.uniform(0.1, 2.0) for _ in range(3)]
            action = AdPlatformAction(allocations=allocations, bids=bids)
            step_result = await env.step(action)

            # Inspect the Observation returned by that step
            print(f"Current Observation: {step_result.observation}")
            print(f"Reward received: {step_result.reward}")
            print(f"Is Done?: {step_result.done}")

            # Inspect internal environment State (if necessary for debugging)
            print("\n--- Checking server-side State ---")
            state = await env.state()

            print(f"Current state: {state.remaining_budget}/{state.total_budget} , {state.step_count}/{state.max_steps}")
            print(f"Total conversions: {state.total_conversions}")

            done = step_result.done

        if step_result.done:
            print("\nEpisode finished successfully!")

if __name__ == "__main__":
    asyncio.run(main())
