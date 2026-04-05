# inference.py

"""
Multi-Task Inference Script
============================
Supports three tasks:
1. budget
2. auction
3. multi_campaign

Emits logs in the mandatory [START], [STEP], [END] format.
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from server.multi_task_environment import MultiTaskEnvironment
from models import BudgetAction, AuctionAction, DynamicAction

# ---------------- CONFIG ----------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME") or "budget"  # budget | auction | multi_campaign
BENCHMARK = os.getenv("BENCHMARK") or "multi_task_env"
IMAGE_NAME = os.getenv("IMAGE_NAME")  # for from_docker_image() if used

MAX_STEPS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0,1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a multi-task environment.
    Depending on the task, you must return an action string that encodes:
    - budget allocations for budget task
    - bid amounts for auction task
    - multi-campaign allocations/bids for multi_campaign task

    Respond with a valid JSON string or Python-like list representing the action(s):
    - Budget: {"allocation": [0.0, 1.2, ...]}
    - Auction: {"allocations": [...], "bids": [...]}
    - Multi-campaign: {"allocations": [...], "bids": [...]}

    Only output the action object. No commentary or extra text.
    """
).strip()


# ---------------- LOGGING ----------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


# ---------------- PROMPT ----------------
def build_user_prompt(step: int, last_action: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last action: {last_action!r}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Provide the next action.
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, last_action: str, last_reward: float, history: List[str]) -> dict:
    prompt = build_user_prompt(step, last_action, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Evaluate safely into dict
        try:
            action = eval(text, {"__builtins__": {}})
            if not isinstance(action, dict):
                action = {}
        except Exception:
            action = {}
        return action
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {}


# ---------------- MAIN LOOP ----------------
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await MultiTaskEnvironment(task=TASK_NAME).from_docker_image(IMAGE_NAME) if IMAGE_NAME else MultiTaskEnvironment(task=TASK_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        last_action = "{}"
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if getattr(result, "done", False):
                break

            action_dict = get_model_action(client, step, last_action, last_reward, history)

            # Wrap in correct Action class
            if TASK_NAME == "budget":
                action_obj = BudgetAction(**action_dict)
            elif TASK_NAME == "auction":
                action_obj = AuctionAction(**action_dict)
            elif TASK_NAME == "multi_campaign":
                action_obj = DynamicAction(**action_dict)
            else:
                raise ValueError(f"Unknown task: {TASK_NAME}")

            result = await env.step(action_obj)
            reward = getattr(result, "reward", 0.0)
            done = getattr(result, "done", False)
            steps_taken = step
            last_action = str(action_dict)
            last_reward = reward
            rewards.append(reward)

            log_step(step, last_action, reward, done, error=None)
            history.append(f"Step {step}: {last_action} -> reward {reward:.2f}")

            if done:
                break

        # compute normalized score
        max_possible_reward = getattr(env, "max_possible_conversions", 1.0)
        score = sum(rewards) / max_possible_reward
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())