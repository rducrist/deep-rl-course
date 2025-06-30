# Vanilla Policy Gradient (REINFORCE) – Assignment

## Objective

Train an agent using a policy network to solve the `CartPole-v1` environment by maximizing expected return via the REINFORCE algorithm.

---

## Milestone 1: Set Up the Environment

**Tasks:**

- [ ] Import necessary libraries: `torch`, `gymnasium`, `numpy`
- [ ] Create the environment using `gym.make("CartPole-v1")`
- [ ] Extract `input_dim` and `output_dim` from observation and action spaces
- [ ] Seed the environment for reproducibility

Hints:
- `env.observation_space.shape[0]`
- `env.action_space.n`

---

## Milestone 2: Build the Policy Network

**Tasks:**

- [ ] Create a class `PolicyNetwork(nn.Module)` with:
  - One or two hidden layers (e.g. 128 units)
  - Final layer outputs action probabilities using `softmax`
- [ ] Forward method should take in a state and output action probabilities

Hints:
- Use `nn.Linear` and `torch.nn.functional.relu`
- Output shape should be `(num_actions,)`

---

## Milestone 3: Write an Action Sampling Function

**Tasks:**

- [ ] Use the policy network to get probabilities
- [ ] Sample from `torch.distributions.Categorical`
- [ ] Return both the action and its log probability (for training)

Hints:
```python
dist = torch.distributions.Categorical(probs)
action = dist.sample()
log_prob = dist.log_prob(action)
