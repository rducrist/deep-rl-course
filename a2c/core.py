import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

env = gym.make("CartPole-v1")
obs = env.reset(seed=42)[0]


# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.actor(x), self.critic(x)


# Action Selection
def select_action(actor_critic: ActorCritic, state):
    state = torch.tensor(state).unsqueeze(0)
    actor_logits, critic_value = actor_critic(state)

    # Actor part
    actor_probs = nn.functional.softmax(actor_logits, dim=-1)

    action_distr = torch.distributions.Categorical(actor_probs)
    action_sample = action_distr.sample()

    action_log_prob = action_distr.log_prob(action_sample)

    # Critic part

    return action_sample, action_log_prob, critic_value.squeeze()


# Generate Episode Rollout
def run_episode(env: gym.Env, actor_critic: ActorCritic):
    done = False
    episode = []
    obs, _ = env.reset()

    while not done:
        # Sample action
        action, action_log_prob, critic_value = select_action(actor_critic, obs)

        # Step the env
        next_obs, reward, term, trunc, _ = env.step(action.item())
        done = term or trunc

        # Store ep data
        episode.append((obs, action, reward, action_log_prob, critic_value))

        obs = next_obs

    return episode


# Compute returns and advantages
def returns_and_advantages(rewards, critic_values, gamma):
    # Returns as in VPG
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    # Normalisation for stability
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - torch.mean(returns)) / returns.std()

    # Advantages = Returns - predicted values. It's the MC way
    advantages = returns - critic_values

    return returns, advantages


# Actor and critic losses
def actor_critic_losses(action_log_probs, advantages, returns, critic_values):
    actor_loss = -torch.sum(action_log_probs.squeeze(-1) * advantages.detach())
    critic_loss = nn.functional.mse_loss(returns, critic_values)

    return actor_loss , critic_loss

# Backpropagate and update the networks
def update_network(optimizer: torch.optim.Adam, loss : torch.Tensor):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def evaluate(env, actor_critic, episodes: int = 5):
    total_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # No exploration — take the most probable action
            state = torch.tensor(obs).float().unsqueeze(0)
            with torch.no_grad():
                logits, _ = actor_critic(state)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Eval Episode {ep+1}: Reward = {episode_reward}")

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"✅ Average Eval Reward over {episodes} episodes: {avg_reward:.2f}")


def main():
    # Init env
    train_env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = train_env.reset()
    input_dim = train_env.observation_space.shape[0]
    output_dim = train_env.action_space.n

    # Init policy network
    actor_critic = ActorCritic(input_dim=input_dim, output_dim=output_dim)

    # Init optimizer
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=0.001)

    # Training loop
    num_episodes = 1000
    gamma = 0.99  
    ep_idx = 0  

    for episode_idx in range(num_episodes):

        episode = run_episode(train_env, actor_critic)

        # Extract data from the episode
        obs, actions, rewards, action_log_probs, critic_values = zip(*episode)

        action_log_probs = torch.stack(action_log_probs)
        critic_values = torch.stack(critic_values).squeeze(-1)  # Ensure shape [T]
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Compute returns and advantages
        returns, advantages = returns_and_advantages(rewards, critic_values, gamma)

        # Compute loss
        actor_loss, critic_loss = actor_critic_losses(action_log_probs, advantages, returns, critic_values)
        total_loss = actor_loss + critic_loss

        # Update policy
        update_network(optimizer, total_loss)

        # Logging
        episode_reward = sum(rewards).item()
        print(f"Episode {episode_idx} — Reward: {episode_reward:.2f} — Actor Loss: {actor_loss.item():.4f} — Critic Loss: {critic_loss.item():.4f}")

        # Optional: evaluate every N episodes
        if episode_idx % 1000 == 0:
            evaluate(eval_env, actor_critic)


if __name__== "__main__":
    main()