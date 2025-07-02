import torch
import gymnasium as gym
from collections import deque
from torch import nn
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)


def select_action(policy_net, state):
    """
    Some notes on the action selection process:
    1. The policy network outputs logits for each action.
    2. We apply softmax to convert logits to probabilities.
    3. We sample an action from the categorical distribution defined by these probabilities.

    ### Why do we get logits and not probabilities?
    It is more convenient as they can take any value. It is easier to optimize.

    ### Why use Categorical?
    The probabilities alone cannot be sampled just like that. We need to create a DISCRETE distribution first.

    At the end we compute the log probabilities as we use them for the REINFORCE trick to compute the gradient.
    """
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    logits = policy_net(state)
    probs = nn.functional.softmax(logits, dim=-1)

    action_distr = torch.distributions.Categorical(probs=probs)
    action_sample = action_distr.sample()

    action_log_prob = action_distr.log_prob(action_sample)

    return action_sample, action_log_prob


def run_episode(env: gym.Env, policy: PolicyNetwork):
    done = False
    episode = []
    obs, _ = env.reset()

    while not done:
        # Sample action
        action, action_log_prob = select_action(policy_net=policy, state=obs)


        # Step the env
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        # Store data
        episode.append((obs, action, reward, action_log_prob))

        obs = next_obs

    return episode


def compute_returns(rewards, gamma):
    """
    Just to clarify my missconception.
    Even though we update the objective function at the end of the episode, we are still computing the return for *every* step.
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    # Normalisation for stability
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - torch.mean(returns)) / returns.std()

    return returns


def evaluate_policy(policy, eval_env, n_eval_episodes=3):
    policy.eval()
    total_rewards = []

    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            logits = policy(obs_tensor)
            action_probs = nn.functional.softmax(logits, dim=-1)
            action = torch.argmax(action_probs).item()  # Greedy action for eval
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward

        total_rewards.append(ep_reward)

    avg_reward = np.mean(total_rewards)
    policy.train()
    return avg_reward

def main():
    # Init env
    train_env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = train_env.reset()
    input_dim = train_env.observation_space.shape[0]
    output_dim = train_env.action_space.n

    # Init policy network
    policy = PolicyNetwork(input_dim=input_dim, output_dim=output_dim)

    # Init optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    # Init vars for logging
    all_rewards = []
    smoothed_rewards = []
    losses = []
    reward_window = deque(maxlen=50)  # moving average window

    # Training loop
    num_episodes = 1000
    gamma = 0.99  
    ep_idx = 0  

    for _ in range(num_episodes):
        # Run the episode
        episode = run_episode(env=train_env, policy=policy)

        # Extract data from the episode
        states, actions, rewards, action_log_probs = zip(*episode)

        action_log_probs = torch.stack(action_log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Compute returns
        returns = compute_returns(rewards, gamma)

        # Compute loss
        action_log_probs = action_log_probs.squeeze(-1)  # shape becomes (N,)
        loss = -torch.sum(action_log_probs * returns) # Episodes can be of different lengths, so we average the loss

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        total_reward = sum(rewards).item()
        reward_window.append(total_reward)
        avg_reward = np.mean(reward_window)

        all_rewards.append(total_reward)
        smoothed_rewards.append(avg_reward)
        losses.append(loss.item())

        ep_idx += 1
        print(f"Episode {ep_idx+1}, Total Reward: {sum(rewards):.2f}, Loss: {loss.item():.6f}")

        # Evaluate every few episodes:
        if ep_idx % 200 == 0:
            avg_eval_reward = evaluate_policy(policy, eval_env, n_eval_episodes=10)
            print(f"Episode {episode}: Average Evaluation Reward: {avg_eval_reward:.2f}")
        eval_env.close()

    train_env.close()
    

    


# '''
# Visualization part
# '''

# import matplotlib.pyplot as plt

# def visualize_action_distribution(probs):
#     plt.bar(range(len(probs)), probs.detach().numpy())
#     plt.xlabel('Action')
#     plt.ylabel('Probability')
#     plt.title('Action Distribution')
#     plt.show()

if __name__ == "__main__":
    main()
    # visualize_action_distribution(probs)  # Uncomment to visualize action distribution