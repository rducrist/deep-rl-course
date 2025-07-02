# Vanilla Policy Gradient Implementation Assignment

This assignment guides you step-by-step to implement a vanilla policy gradient (REINFORCE) agent from scratch using PyTorch and Gymnasium.

---

## Milestone 1: Setup Environment & Policy Network

- Initialize the environment (e.g., `"CartPole-v1"`).
- Create a `PolicyNetwork` class inheriting from `torch.nn.Module`.
- The network should take state observations as input and output raw action logits.
- Use two hidden layers with 128 units each and ReLU activations.

---

## Milestone 2: Action Selection

- Implement a function `select_action(policy_net, state)` that:
  - Takes a state observation and policy network as inputs.
  - Runs the state through the network to get logits.
  - Applies softmax to convert logits to action probabilities.
  - Samples an action from the categorical distribution.
  - Returns the selected action and the log probability of that action.

---

## Milestone 3: Generate Episode Rollouts

- Implement a function to run an episode:
  - Reset the environment.
  - Use the `select_action` function to choose actions.
  - Collect `(state, action, reward, log_prob)` tuples for each step.
  - Continue until the episode ends (done).
- Return the collected episode data.

---

## Milestone 4: Compute Returns

- Implement a function to compute discounted returns from rewards:
  - Use a discount factor `gamma` (e.g., 0.99).
  - Returns should be normalized (mean 0, std 1) to improve training stability.

---

## Milestone 5: Update Policy Network

- Implement the policy gradient loss:
  - For each step, multiply the negative log probability of the action by the corresponding discounted return.
  - Average the loss over the episode.
- Use an optimizer (e.g., Adam) to update the policy network parameters via backpropagation.

---

## Milestone 6: Training Loop

- Combine the above components into a training loop that:
  - Runs episodes.
  - Collects rollout data.
  - Computes returns.
  - Updates the policy network.
  - Tracks and prints total episode rewards.
- Train for a specified number of episodes (e.g., 1000).

---

## Bonus

- Plot the episode rewards over time.
- Experiment with different network architectures and learning rates.
- Add entropy regularization to encourage exploration.

---

## Tips

- Use `torch.distributions.Categorical` for sampling actions and computing log probabilities.
- Keep tensor shapes consistent.
- Use `torch.no_grad()` when running the environment to avoid tracking gradients.
- Normalize returns to improve convergence.

---

By completing this assignment, you will gain a solid understanding of the policy gradient method and be able to implement a vanilla REINFORCE algorithm from scratch!

---

Good luck! 🚀
