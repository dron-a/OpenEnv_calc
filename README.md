# Ad Platform Environment

A multi-task **Reinforcement Learning environment** built on the [OpenEnv](https://github.com/openenv) framework. It simulates a real-world digital advertising platform where an agent must allocate budgets, set bids, and maximize ad conversions across multiple campaigns.

---

## Environment Overview

The agent interacts with an ad platform over a fixed-length episode (30 steps). At each step it allocates budget and sets bids across **3 campaigns**. The environment tracks spending, computes conversions based on win probability against adaptive competitors, and returns shaped rewards.

Three tasks are available with increasing difficulty:

| Task | Description | Difficulty |
|---|---|---|
| `budget` | Allocate a fixed budget across campaigns to maximize conversions. No bidding. | Easy |
| `auction` | Compete against adaptive bidders. Must bid and allocate to win auctions. | Medium |
| `multi_campaign` | Full complexity — dynamic conversion rates, market events, seasonality. | Hard |

---

## Project Structure

```text
.
├── models.py                    # Pydantic schemas: Action, Observation, State
├── grader.py                    # Independent evaluation scores for all 3 tasks
├── inference.py                 # Baseline LLM agent inference script
├── client.py                    # OpenEnv async client wrapper
├── example_usage.py             # Example episode loop using the client
├── pyproject.toml               # Project metadata
├── openenv.yaml                 # OpenEnv spec configuration
├── Dockerfile                   # Container definition
└── server/
    ├── app.py                   # FastAPI entry point and session factory
    ├── environment.py           # Unified AdPlatformEnvironment class
    ├── requirements.txt         # Python dependencies
    └── tasks/
        ├── task1_budget.py      # Budget task logic
        ├── task2_auction.py     # Auction task logic
        └── task3_multi_campaign.py  # Multi-campaign task logic
```

---

## Getting Started

### Build and Run

```bash
docker build -t ad-platform:latest .
docker run -it --rm -p 8000:8000 ad-platform:latest
```

The server starts on `http://localhost:8000`. Interactive API docs: `http://localhost:8000/docs`

To select a task, set the `TASK` environment variable:

```bash
docker run -it --rm -p 8000:8000 -e TASK=auction ad-platform:latest
```

---

## API

### Reset (POST `/reset`)

Start a new episode.

```json
{}
```

Response:
```json
{
  "observation": {
    "step": 0,
    "remaining_budget": 1000.0,
    "campaign_performance": [0.05, 0.03, 0.02],
    "competitor_bids": [],
    "reward": 0.0,
    "done": false
  },
  "reward": null,
  "done": false
}
```

### Step (POST `/step`)

Submit an action.

```json
{
  "action": {
    "allocations": [100.0, 50.0, 30.0],
    "bids": [0.6, 0.5, 0.4]
  }
}
```

Response:
```json
{
  "observation": {
    "step": 1,
    "remaining_budget": 820.0,
    "campaign_performance": [0.05, 0.03, 0.02],
    "competitor_bids": [0.51, 0.42, 0.31],
    "reward": 3.21,
    "done": false
  },
  "reward": 3.21,
  "done": false
}
```

### State (GET `/state`)

Inspect full internal state.

---

## Reward Design

Each task uses a shaped reward:

```
reward = delayed_conversions - spend_penalty - illegal_action_penalty - carryover_penalty
```

- **delayed_conversions**: conversions from the previous step (1-step delay)
- **spend_penalty**: penalizes spending too large a fraction of total budget in one step
- **illegal_action_penalty**: penalizes negative or pacing-violating allocations
- **carryover_penalty**: penalizes aggressive early spending (decreases as episode progresses)

---

## Grader

Independent evaluation functions in `grader.py` score episodes on a `[0, 1]` scale:

```python
from grader import compute_score, compute_auction_score, compute_multi_campaign_score

# After an episode ends:
score = compute_score(env._state)                      # budget task
result = compute_auction_score(env._state)             # auction task
result = compute_multi_campaign_score(env._state)      # multi_campaign task
print(result["final_score"])
```

---

## Running the Baseline Agent

```bash
export API_KEY=your_hf_token
export TASK_NAME=auction
python inference.py
```
