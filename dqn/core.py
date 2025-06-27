import gymnasium as gym
import numpy as np

from torch import nn
import torch

import random
import matplotlib.pyplot as plt

# === Models ===
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)

    
# === Replay Buffer ===
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    

# === Policy ===
def epsilon_greedy_action(q_values, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))  # Explore: random action
    else:
        return np.argmax(q_values)  # Exploit: best action based on Q-values
    

# === Training Step ===    
def train_step(q_network : QNetwork, target_network : QNetwork, buffer : ReplayBuffer, optimizer : torch.optim.Adam, batch_size=32, gamma=0.99):
    
    samples = buffer.sample(batch_size)

    states, actions, rewards, next_states, dones = zip(*samples)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Compute Q-values for current states
    q_values = q_network(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q-values
    with torch.no_grad():
        # Get Q-values for next states from target network
        target_q_values = target_network(next_states)

        # Classical implementation uses max over actions
        # q_target = rewards + (1 - dones) * gamma * torch.max(target_q_values, dim=1)[0]

        # Double DQN uses the current Q-network to select actions
        next_actions = torch.argmax(q_network(next_states), dim=1)
        q_target = rewards + (1 - dones) * gamma * target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

    # Compute loss
    loss = nn.functional.mse_loss(q_values, q_target)

    # Perform backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# === Evaluation ===
def evaluate_agent(env : gym.Env, q_network : QNetwork, num_episodes=10):
    q_network.eval()
    total_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            q_values = q_network(obs_tensor)
            action = torch.argmax(q_values).item()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        print(f"Eval Episode {ep+1}: Reward = {ep_reward}")
        total_rewards.append(ep_reward)

    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards)}")

# === Visualization ===
def moving_average(data, window_size=10):
    data = np.array(data)
    if len(data) < window_size:
        return data, data
    averages = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    stds = np.array([np.std(data[i-window_size:i]) if i >= window_size else 0 for i in range(window_size, len(data)+1)])
    return averages, stds

# === Main Execution ===    
def main():
    # Create environment
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset(seed=42)

    # Initialize networks and optimizer
    q_network = QNetwork(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)
    target_network = QNetwork(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-3)

    # Initialize replay buffer
    buffer = ReplayBuffer(capacity=10000)

    num_episodes = 1000
    episode_rewards = []
    episode_losses = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0
        done = False

        while not done:
            # Get action from epsilon-greedy policy
            q_values = q_network(torch.tensor(obs, dtype=torch.float32))

            # Epsilon should be decaying
            epsilon = max(0.01, 0.1* (0.995 ** episode))
            action = epsilon_greedy_action(q_values.detach().numpy(), epsilon=epsilon)

            # Step the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            buffer.add((obs, action, reward, next_obs, done))

            obs = next_obs
            ep_reward += reward

        episode_rewards.append(ep_reward)
        print(f"Episode {episode + 1}: Reward = {ep_reward}")
        # Train the Q-network
        if len(buffer) > 32:
            episode_losses.append(train_step(q_network, target_network, buffer, optimizer, batch_size=32))
        # Update target network
        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())

    torch.save(q_network.state_dict(), "./dqn/model/dqn_cartpole.pth")
    print("Training completed.")

    # evaluate_agent(eval_env, q_network, num_episodes=10)

    # Compute smoothed rewards and losses
    avg_rewards, std_rewards = moving_average(episode_rewards, window_size=50)
    avg_losses, std_losses = moving_average(episode_losses, window_size=50)

    # Create plots
    plt.figure(figsize=(12, 5))

    # Plot rewards
    plt.subplot(1, 2, 1)
    x_rewards = np.arange(len(avg_rewards))
    plt.plot(x_rewards, avg_rewards, label='Smoothed Reward')
    plt.fill_between(x_rewards, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.3, label='±1 std')
    plt.title("Episode Rewards (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    # Plot losses
    plt.subplot(1, 2, 2)
    x_losses = np.arange(len(avg_losses))
    plt.plot(x_losses, avg_losses, label='Smoothed Loss', color='orange')
    plt.fill_between(x_losses, avg_losses - std_losses, avg_losses + std_losses, alpha=0.3, label='±1 std', color='orange')
    plt.title("Episode Loss (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("./dqn/plots/training_metrics_smoothed.png")
    plt.show()



if __name__ == "__main__":
    main()