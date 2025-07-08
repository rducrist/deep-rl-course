import torch
import gymnasium as gym
from collections import deque
from torch import nn
import numpy as np
import random

train_env = gym.make("Pendulum-v1")
eval_env = gym.make("Pendulum-v1", render_mode="human")
obs_dim = train_env.observation_space.shape[0]
act_dim = train_env.action_space.shape[0]
action_high = train_env.action_space.high
action_low = train_env.action_space.low
obs = train_env.reset(seed=42)[0]


# Create actor critic network
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, action_bound):
        super(Actor, self).__init__()
        # No idea what it does but it works
        self.register_buffer(
            "action_bound", torch.tensor(action_bound, dtype=torch.float32)
        )

        self.fc = nn.Linear(input_dim, 128)
        # Because the policy output is a probability distribution over the actions
        self.out = nn.Linear(128, output_dim)

    def forward(self, state):
        x = torch.relu(self.fc(state))
        x = torch.tanh(self.out(x))
        return x * self.action_bound


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, state, action):
        # Has to be a list []
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc(x))
        x = self.out(x)
        return x


class ReplayBuffer:
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


def select_action(state, policy: Actor, noise_scale=0.3):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = policy(state)

    # Exploration noise
    noise = torch.randn(len(action)) * noise_scale
    action += noise
    action = torch.clamp(action, action_low[0], action_high[0])

    return action.squeeze(0).numpy()


def critic_update(
    rewards,
    dones,
    gamma,
    target_q_values: torch.Tensor,
    q_values: torch.Tensor,
    optimizer: torch.optim.Adam,
):
    # Compute the target Q value: y = r + (1 - done) * γ * Q_target(s', a')
    # Detach target Q values to not have gradients
    q_target = rewards + (1 - dones) * gamma * target_q_values.detach()

    # Minimze the MSE loss
    critic_loss = nn.functional.mse_loss(q_target, q_values)

    # Perform backpropagation
    optimizer.zero_grad()
    critic_loss.backward()
    optimizer.step()

    return critic_loss.item()


def actor_update(states, actor: Actor, critic: Critic, optimizer: torch.optim.Adam):
    predicted_actions = actor(states)
    # minimize -Q(s, pi(s))
    target_q_values = critic(states, predicted_actions)
    actor_loss = -target_q_values.mean()

    optimizer.zero_grad()
    actor_loss.backward()
    optimizer.step()

    return actor_loss.item()


def soft_update(target_net: nn.Module, source_net: nn.Module, tau=0.005):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def evaluate_policy(env, actor: Actor, episodes=3):
    actor.eval()
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor(state)
            action = action.squeeze(0).numpy()
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += reward
        print(f"Evaluation Episode {ep + 1}: Reward = {total_reward:.2f}")
    actor.train()


def main():
    # Init networks
    actor = Actor(obs_dim, act_dim, action_high)
    actor_target = Actor(obs_dim, act_dim, action_high)

    critic = Critic(obs_dim + act_dim)
    critic_target = Critic(obs_dim + act_dim)

    # Sync target networks
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)

    buffer = ReplayBuffer(capacity=10000)

    num_episodes = 10000
    gamma = 0.99
    batch_size = 64
    eval_interval = 1000

    for ep_idx in range(1, num_episodes + 1):
        obs, _ = train_env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = select_action(obs, actor)
            next_obs, reward, term, trunc, _ = train_env.step(action)
            done = term or trunc
            buffer.add((obs, action, reward, next_obs, done))
            obs = next_obs
            episode_reward += reward

        if len(buffer) > batch_size:
            samples = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*samples)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            next_actions = actor_target(next_states)
            target_q_values = critic_target(next_states, next_actions)
            current_q_values = critic(states, actions)

            critic_loss = critic_update(
                rewards,
                dones,
                gamma,
                target_q_values,
                current_q_values,
                critic_optimizer,
            )
            actor_loss = actor_update(states, actor, critic, actor_optimizer)

            soft_update(actor_target, actor)
            soft_update(critic_target, critic)

            if ep_idx % 10 == 0:
                print(
                    f"Episode {ep_idx} | Reward: {episode_reward:.2f} | Critic Loss: {critic_loss:.4f} | Actor Loss: {actor_loss:.4f}"
                )

        # Evaluation
        if ep_idx % eval_interval == 0:
            print(f"\n--- Evaluating policy at Episode {ep_idx} ---")
            evaluate_policy(eval_env, actor, episodes=3)
            print("--- Evaluation complete ---\n")


if __name__ == "__main__":
    main()
