import random
import numpy as np
from collections import defaultdict
import pickle
import asyncio
from client import AdEnvClient
from models import AdSpendAction

# --- 1. Setup & Config ---
PLATFORMS = ['email', 'facebook', 'google', 'instagram', 'whatsapp', 'youtube']
SPEND_TIERS = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]
# Total 36 discrete actions (6 platforms * 6 spend tiers)
ACTIONS = [(p, s) for p in PLATFORMS for s in SPEND_TIERS]
NUM_ACTIONS = len(ACTIONS)

def discretize_state(obs):
    """Converts continuous observations to discrete state keys."""
    # Ensure budget is treated as a float and handle potential precision issues
    safe_budget = max(0.0, float(obs.remaining_budget))
    budget_bucket = int(safe_budget // 100)

    # Days are already discrete in the environment logic (0-30)
    days_left = max(0, int(obs.days_remaining))

    # Bucket leads to keep the state space size optimal for Q-learning
    prev_leads = int(obs.previous_day_leads)
    if prev_leads < 5:
        leads_bucket = "Low"
    elif prev_leads < 15:
        leads_bucket = "Med"
    else:
        leads_bucket = "High"

    return f"{budget_bucket}_{days_left}_{leads_bucket}"


# --- 2. Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.998, min_epsilon=0.01):
        self.q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float64))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_action_index(self, state_key):
        """Robustly selects the best action index from the Q-table."""
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)

        q_values = self.q_table[state_key]
        max_val = np.max(q_values)

        # np.flatnonzero returns a single ndarray directly,
        # avoiding the tuple-unpacking issues of np.where.
        best_indices = np.flatnonzero(q_values == max_val).tolist()

        # Select one index randomly among ties to maintain policy balance
        return int(random.choice(best_indices))

    def update(self, state_key, action_idx, reward, next_state_key, done):
        """Update the Q-value for the (state, action) pair using the Bellman equation."""
        best_next_q = 0.0 if done else float(np.max(self.q_table[next_state_key]))
        current_q = float(self.q_table[state_key][action_idx])

        # Q-learning formula (Temporal Difference)
        new_q = current_q + self.alpha * (reward + (self.gamma * best_next_q) - current_q)
        self.q_table[state_key][action_idx] = new_q

    def decay_epsilon(self):
        """Decrease the probability of taking a random action as training progresses."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# --- 3. Async Training Loop ---
async def train_tabular_rl():
    # Initialize the client to communicate with the Docker environment
    env = AdEnvClient(base_url="http://localhost:8000")
    agent = QLearningAgent()
    epochs = 3000

    print(f"Initializing Tabular Q-Learning...")
    print(f"Total Discrete Actions: {NUM_ACTIONS} | Training for {epochs} epochs")

    try:
        for epoch in range(1, epochs + 1):
            # 1. Reset Environment via API (Async)
            reset_result = await env.reset()
            # Support both direct and wrapped observation returns
            obs = reset_result.observation if hasattr(reset_result, 'observation') else reset_result

            state_key = discretize_state(obs)
            total_reward = 0.0
            done = False

            # 2. Episode Loop
            while not done:
                # Select action using epsilon-greedy policy
                action_idx = agent.get_action_index(state_key)
                platform, spend = ACTIONS[action_idx]

                # Execute action via API call (Async)
                action_model = AdSpendAction(platform=platform, spend_amount=float(spend))
                step_result = await env.step(action_model)

                next_obs = step_result.observation
                reward = float(step_result.reward or 0.0)
                done = bool(step_result.done)

                # Immediate termination if budget is depleted
                if next_obs.remaining_budget <= 0:
                    done = True

                next_state_key = discretize_state(next_obs)

                # 3. Update the agent's knowledge base
                agent.update(state_key, action_idx, reward, next_state_key, done)

                state_key = next_state_key
                total_reward += reward

            # Post-episode epsilon decay
            agent.decay_epsilon()

            # Console progress logging
            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch:04d} | Eps: {agent.epsilon:.3f} | Total Leads: {total_reward:5.1f} | Final Budget: ${next_obs.remaining_budget:7.2f}")

    except Exception as e:
        print(f"A critical error occurred during training: {e}")
        raise e
    finally:
        # Persistence: save the learned Q-table for future evaluation
        with open("q_table.pkl", "wb") as f:
            pickle.dump(dict(agent.q_table), f)
        print("Training session finished. Q-Table saved to q_table.pkl")


if __name__ == "__main__":
    # Run the asynchronous loop
    asyncio.run(train_tabular_rl())