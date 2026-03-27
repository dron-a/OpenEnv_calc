# Calculator Target Sum Environment

This project is a **Reinforcement Learning (RL) environment** built using the **OpenEnv** framework. It simulates a calculator-based target sum task where an agent must navigate an internal state to reach a specific goal.

## 🎯 Environment Objective

The agent's goal is to reach a sum **equal to or greater than 10**.

* The target sum is **not** provided to the agent at the start.
* The agent must use `step` and `state` calls to identify the current sum and determine the optimal sequence of actions.
* The environment returns `done=true` once the target condition is met.

---

## 📁 Project Structure

Please focus on the following core files. Other files in the repository are related to development, debugging, or CI/CD and can be ignored.

```text
.
├── models.py                # Pydantic schemas for Action, Observation, and State
├── pyproject.toml           # Project metadata and dependencies
├── .dockerignore            # Files to exclude from Docker build
├── .gitignore               # Files to exclude from Git
└── server/
    ├── app.py               # FastAPI server entry point and environment factory
    ├── environment.py       # Core RL logic and CalculatorEnvironment class
    └── requirements.txt     # Python dependencies for the server
```

---

## 🚀 Getting Started

### Build and Run

After pulling or forking the master branch, use the following commands to containerize and launch the environment.

**Build the image:**

```bash
docker build -t calc:latest .
```

**Run the container:**

```bash
docker run -it --rm -p 8000:8000 calc:latest
```

---

## 📘 API Documentation

Once running, the interactive API documentation is available at: **`http://0.0.0.0:8000/docs`**

---

## 🛠 Interaction Protocol

Every trial episode must follow this sequence:

1. **Reset:** Call the `/reset` endpoint at the start of every new episode.
2. **Loop:** Iteratively call `/step` and `/state` to navigate the environment.
3. **Terminate:** Continue until the response returns `"done": true`.

### 1. Reset Environment (POST)

Initializes a new episode.

* **URL:** `http://0.0.0.0:8000/reset`
* **Example Body:**

    ```json
    {
      "episode_id": "episode-001",
      "seed": 42
    }
    ```

* **Example Response:**

    ```json
    {
      "observation": { "current_value": 0 },
      "reward": null,
      "done": false
    }
    ```

### 2. Take a Step (POST)

Perform an action. You can use commands `"add"` or `"sub"` (subtract).

* **URL:** `http://0.0.0.0:8000/step`
* **Example Body:**

    ```json
    {
      "action": {
        "command": "add",
        "amount": 20
      }
    }
    ```

* **Example Response:**

    ```json
    {
      "observation": { "current_value": 20 },
      "reward": 0.0,
      "done": true
    }
    ```

### 3. Get Internal State (GET)

View the current internal state.

* **URL:** `http://0.0.0.0:8000/state`
* **Example Response:**

    ```json
    {
      "episode_id": "episode-001",
      "step_count": 1,
      "current_sum": 20,
    }
    ```

---

## 🔍 Development Note

Explore `models.py` to see how the Pydantic schemas define the structure of the API responses. The strict typing ensures that the `CalcAction`, `CalcObservation`, and `CalcState` models are consistently reflected across all endpoints.

---

## 🧩 Understanding the Code

* The **API request/response schema** is defined in:

  ```bash
  models.py
  ```

* These schemas are directly reflected in the responses of `/reset`, `/step`, and `/state`.

* Core logic of the RL environment is implemented in:

  ```bash
  server/environment.py
  ```

* API routing and server setup:

  ```bash
  server/app.py
  ```

---

## 🎯 Summary

* RL environment where agent must infer a hidden target ≥ 10
* Interaction via `reset → state → step`
* Simple arithmetic action space (`add`, `sub`)
* Fully containerized using Docker
* Schema-driven API design via `models.py`

---

## 💡 Notes

* Designed for RL experimentation and evaluation
* Can be extended with:
  * More complex reward shaping
  * Larger action space
  * Multi-step reasoning policies