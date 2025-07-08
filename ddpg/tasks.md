
# Deep Deterministic Policy Gradient (DDPG): Milestone Guide

> Based on: [Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971)

DDPG is an **off-policy actor-critic algorithm** designed for **continuous action spaces**. It combines the deterministic policy gradient with Q-learning techniques and uses **target networks** and a **replay buffer** to stabilize learning.

---

## Milestone Overview

---

### Milestone 1: Environment Setup

- Use a continuous control task such as:
  - `Pendulum-v1`
  - `LunarLanderContinuous-v2`
- Inspect observation and action spaces:
  ```python
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]
  action_high = env.action_space.high
  action_low = env.action_space.low
  ```

---

### Milestone 2: Define Networks

- **Actor Network**  
  Maps state to action:
  ```math
  \mu(s | \theta^\mu) \rightarrow a
  ```
  Output must be bounded within the action space (e.g., using `tanh`).

- **Critic Network**  
  Maps (state, action) pair to Q-value:
  ```math
  Q(s, a | \theta^Q) \rightarrow \mathbb{R}
  ```

- Create **target networks** for both actor and critic:
  - \( \mu' \), \( Q' \)

---

### Milestone 3: Replay Buffer

- Store tuples:  
  \( (s_t, a_t, r_t, s_{t+1}, d_t) \)
- Sample random mini-batches during training for off-policy updates.

---

### Milestone 4: Action Selection with Exploration

- Actor outputs deterministic action:  
  ```math
  a_t = \mu(s_t)
  ```
- Add exploration noise:
  ```math
  a_t = \mu(s_t) + \mathcal{N}_t
  ```
  where \( \mathcal{N}_t \) is either:
  - Gaussian noise
  - Ornstein-Uhlenbeck (OU) process

---

### Milestone 5: Critic Update

- Compute the **target Q-value**:
  ```math
  y_t = r_t + \gamma (1 - d_t) Q'(s_{t+1}, \mu'(s_{t+1}) | \theta^{Q'})
  ```
- Minimize the **MSE loss**:
  ```math
  L(\theta^Q) = \frac{1}{N} \sum (Q(s_t, a_t | \theta^Q) - y_t)^2
  ```

---

### Milestone 6: Actor Update

- Use the **deterministic policy gradient**:
  ```math
  \nabla_{\theta^\mu} J \approx \frac{1}{N} \sum \nabla_a Q(s, a | \theta^Q) \big|_{a = \mu(s)} \cdot \nabla_{\theta^\mu} \mu(s | \theta^\mu)
  ```

- Backpropagate through the critic to optimize the actor.

---

### Milestone 7: Target Network Updates

- Soft update target networks:
  ```math
  \theta' \leftarrow \tau \theta + (1 - \tau) \theta'
  ```
  Typically \( \tau = 0.005 \)

---

### Milestone 8: Training Loop

1. Initialize actor, critic, and their target networks.
2. For each episode:
   - Collect data using noisy actions.
   - Store transitions in replay buffer.
   - Sample random mini-batch.
   - Update critic (Milestone 5).
   - Update actor (Milestone 6).
   - Update target networks (Milestone 7).
3. Track and log rewards per episode.

---

### Milestone 9: Evaluation

- Disable noise:  
  ```math
  a_t = \mu(s_t)
  ```
- Run multiple evaluation episodes.
- Report average reward.

---

## 🔢 Summary of Key Equations

| Component       | Equation                                                                                   |
|----------------|---------------------------------------------------------------------------------------------|
| Critic Target   | \( y_t = r_t + \gamma (1 - d_t) Q'(s_{t+1}, \mu'(s_{t+1})) \)                               |
| Critic Loss     | \( \mathcal{L} = \frac{1}{N} \sum (Q(s_t, a_t) - y_t)^2 \)                                  |
| Actor Gradient  | \( \nabla_{\theta^\mu} J = \nabla_a Q(s, a) \cdot \nabla_{\theta^\mu} \mu(s) \)            |
| Target Update   | \( \theta' \leftarrow \tau \theta + (1 - \tau) \theta' \)                                  |

---

Ready for the code scaffold when you are 🚀