# Assignment: Implement Advantage Actor-Critic (A2C) from Scratch

## Overview

In this assignment, you will implement the Advantage Actor-Critic (A2C) algorithm from scratch using PyTorch and Gym. You will progressively build the components of A2C including the actor, the critic, rollout generation, advantage computation, and model training.

---

## Milestones

### Milestone 1: Environment Setup
- Initialize a Gym environment (e.g., `CartPole-v1`).
- Extract input and output dimensions from the observation and action spaces.
- Set seeds for reproducibility.

---

### Milestone 2: Build the Actor-Critic Network
- Create a neural network with:
  - A shared body (common layers).
  - Separate output heads for:
    - Action probabilities (actor).
    - State value (critic).

---

### Milestone 3: Action Selection
- Implement a function that:
  - Takes a state.
  - Returns a sampled action and its log probability.
  - Also returns the state value predicted by the critic.

---

### Milestone 4: Generate Episode Rollouts
- Implement a function to run a single episode:
  - Collect `(state, action, reward, log_prob, value)` tuples at each step.
  - Return the collected data after the episode ends.

---

### Milestone 5: Compute Returns and Advantages
- Compute discounted returns for the episode using a discount factor `gamma`.
- Compute advantages using the difference between returns and predicted values.

---

### Milestone 6: Compute Actor and Critic Losses
- Actor loss: Use advantage-weighted policy gradient loss.
- Critic loss: Use mean squared error between predicted values and returns.
- Combine both losses (optionally weighted).

---

### Milestone 7: Update the Network
- Use an optimizer (e.g., Adam) to update the network parameters using the combined loss.
- Perform the update after each episode.

---

### Milestone 8: Training Loop
- Loop over multiple episodes:
  - Collect rollouts.
  - Compute returns and advantages.
  - Update the network.
  - Track total episode rewards and losses.

---

### Milestone 9: Logging and Evaluation
- Log:
  - Total episode rewards.
  - Actor and critic losses.
  - Average rewards (smoothed).
- After training, evaluate the agent’s performance using a deterministic policy.

---

### Optional Milestone 10: Improvements
- Add entropy bonus to the actor loss to encourage exploration.
- Use Generalized Advantage Estimation (GAE).
- Experiment with batch updates over multiple episodes.

---

## Notes

- Use torch tensors and avoid numpy conversions inside the model.
- Make sure to detach gradients where necessary to avoid backprop through time.
- Normalize advantages for training stability.

