import asyncio
from client import CalcEnv
from models import CalcAction

async def main():
    # Connect to the running OpenEnv server.
    base_url = "ws://localhost:5000"
    
    print(f"Connecting to environment at {base_url}...")

    async with CalcEnv(base_url=base_url) as env:
        print("Successfully connected!")

        # 1. Reset the environment to start a new episode
        print("\n--- Resetting Environment ---")
        reset_result = await env.reset()
        print(f"Initial Observation: current_value = {reset_result.observation.current_value}")
        print(f"Is Done?: {reset_result.done}")

        done = False

        while not done:
            #  Take a specific action
            amount1 = input("Enter an amount: ")
            action1 = CalcAction(command="add", amount=amount1)
            step_result = await env.step(action1)

            # Inspect the Observation returned by that step
            print(f"Value seen: {step_result.observation.current_value}")
            print(f"Reward received: {step_result.reward}")
            print(f"Is Done?: {step_result.done}")

            # 3. Inspect internal environment State (if necessary for debugging)
            print("\n--- Checking server-side State ---")
            state = await env.state()

            print(f"Internal sum stored on server: {state.current_sum}")
            print(f"Total steps taken globally: {state.step_count}")

            done = step_result.done

        if step_result.done:
            print("\nEpisode finished successfully!")

if __name__ == "__main__":
    asyncio.run(main())
