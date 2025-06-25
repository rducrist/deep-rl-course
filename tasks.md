### Milestone 1 — Environment Setup

Objective: Become familiar with your RL environment and its API.

Tasks:

    Select a Gym environment (e.g. CartPole-v1).

    Write code to:

        Reset the environment.

        Step through it using random actions.

        Render observations.

    Log total rewards per episode.

Deliverable: A script that runs episodes and prints total reward.

### Milestone 2 — Q-Network Architecture

Objective: Implement a neural network that approximates Q-values.

Tasks:

    Define a feedforward network taking in states, outputting Q-values per action.

    Ensure correct dimensions for output.

    Write a forward method to compute Q(s, a).

Deliverable: A QNetwork class and a test that passes in dummy states and prints Q-values.

### Milestone 3 — Action Selection

Objective: Implement ε-greedy exploration strategy.

Tasks:

    Define a function to choose an action based on Q-values with ε-greedy.

    Include ε decay schedule.

Deliverable: Code that logs selected actions and current ε value per step.
Milestone 4 — Experience Replay Buffer

Objective: Implement a replay buffer to store and sample past transitions.

Tasks:

    Write a ReplayBuffer class with:

        add(state, action, reward, next_state, done)

        sample(batch_size)

    Validate shapes and random sampling.

Deliverable: Unit tests showing the buffer adds and samples transitions correctly.
Milestone 5 — Training Logic

Objective: Implement one gradient step using sampled batch.

Tasks:

    Sample a batch from the buffer.

    Compute:

        Q(s, a)

        Target: r + γ * max_a' Q(next_s, a') (using the target network)

    Backpropagate MSE loss and update Q-network.

Deliverable: A function train_step() that can be called repeatedly in a loop.
Milestone 6 — Target Network

Objective: Stabilize training by using a fixed target network.

Tasks:

    Create a separate target Q-network.

    Implement periodic weight updates (e.g., every N steps).

    Ensure it’s used only for computing the target.

Deliverable: Code to synchronize weights and verify it updates correctly.
Milestone 7 — Training Loop

Objective: Combine all components into a working training pipeline.

Tasks:

    Run multiple episodes.

    Collect experience, train the network.

    Decay ε, log metrics (reward, loss).

    Add checkpointing if needed.

Deliverable: Script that trains a DQN agent and logs reward curves over episodes.
Milestone 8 — Evaluation and Visualization

Objective: Evaluate your trained agent’s performance.

Tasks:

    Run evaluation episodes (no exploration).

    Plot reward over time.

    Optionally: use TensorBoard or matplotlib.

Deliverable: A plot showing learning progress and an evaluation report.
