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
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from server.environment import AdPlatformEnvironment
from models import AdPlatformAction

# ---------------- CONFIG ----------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME") or "budget"  # budget | auction | multi_campaign
BENCHMARK = os.getenv("BENCHMARK") or "ad_platform_env"
IMAGE_NAME = os.getenv("IMAGE_NAME")  # for from_docker_image() if used

MAX_STEPS = 30
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0,1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a multi-task ad platform environment.
    Depending on the task, you must allocate budgets and set bids across 3 campaigns.

    Respond with a valid JSON object only — no commentary or extra text.

    Action format (all tasks):
      {"allocations": [<float>, <float>, <float>], "bids": [<float>, <float>, <float>]}

    Guidelines:
    - allocations: budget to spend per campaign this step (non-negative floats)
    - bids: bid price per campaign (required for auction/multi_campaign; use [0,0,0] for budget task)
    - Stay within the remaining budget; do not overspend in early steps
    - Prefer campaigns with higher conversion rates
    - For auction/multi_campaign: bid above competitor bids to win auctions
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
def build_user_prompt(step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Remaining budget: {obs.get('remaining_budget', 'unknown')}
        Campaign conversion rates: {obs.get('campaign_performance', [])}
        Competitor bids: {obs.get('competitor_bids', [])}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Provide the next action.
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, obs: dict, last_reward: float, history: List[str]) -> dict:
    prompt = build_user_prompt(step, obs, last_reward, history)
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
        try:
            action = json.loads(text)
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

    if IMAGE_NAME:
        env = await AdPlatformEnvironment.from_docker_image(IMAGE_NAME)
    else:
        env = AdPlatformEnvironment(task=TASK_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()
        last_reward = 0.0
        obs = result.model_dump() if hasattr(result, "model_dump") else {}

        for step in range(1, MAX_STEPS + 1):
            if getattr(result, "done", False):
                break

            action_dict = get_model_action(client, step, obs, last_reward, history)

            # Provide default bids for budget task if model omits them
            if "allocations" not in action_dict:
                action_dict["allocations"] = [0.0, 0.0, 0.0]
            if "bids" not in action_dict:
                action_dict["bids"] = []

            try:
                action_obj = AdPlatformAction(**action_dict)
            except Exception as exc:
                print(f"[DEBUG] Invalid action, using zeros: {exc}", flush=True)
                action_obj = AdPlatformAction(allocations=[0.0, 0.0, 0.0], bids=[])

            result = env.step(action_obj)
            reward = getattr(result, "reward", 0.0) or 0.0
            done = getattr(result, "done", False)
            obs = result.model_dump() if hasattr(result, "model_dump") else {}

            steps_taken = step
            last_reward = reward
            rewards.append(reward)

            log_step(step, str(action_dict), reward, done, error=None)
            history.append(f"Step {step}: {action_dict} -> reward {reward:.2f}")

            if done:
                break

        # Compute normalized score against theoretical max conversions
        max_possible_reward = getattr(env, "max_possible_conversions", 1.0) or 1.0
        score = sum(rewards) / max_possible_reward
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            if hasattr(env, "close"):
                await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
