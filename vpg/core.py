import torch
import gymnasium as gym

from torch import nn


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
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store data
        episode.append((obs, action, reward, action_log_prob))

        obs = next_obs


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
