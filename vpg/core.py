import torch 
import gymnasium as gym 
import numpy as np
from torch import nn

env = gym.make("CartPole-v1", render_mode="human")
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

obs, info = env.reset(seed=42)

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
    logits = policy_net(state)
    probs = nn.functional.softmax(logits, dim=-1)

    action_distr = torch.distributions.Categorical(probs=probs)
    action_sample = action_distr.sample()

    action_log_prob = torch.distributions.log

    return action_sample, 
    
