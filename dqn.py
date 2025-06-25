import gymnasium as gym
import numpy as np

from torch import nn
import torch


# create NN that approximates Q-values
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

q_ntwork = QNetwork(input_dim=4, output_dim=2)
example_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)  # shape: (1, 4)
output = q_ntwork.forward(example_input)
print(output)

# # create environment
# env = gym.make("CartPole-v1", render_mode="human")
# obs, info = env.reset(seed=42)

# episode_rewards = []
# ep_reward = 0
# num_episodes = 0

# for _ in range(1000):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     ep_reward += reward

#     if terminated or truncated:
#         episode_rewards.append(ep_reward)
#         print(f"Episode {num_episodes + 1} - Reward: {ep_reward}")
#         ep_reward = 0
#         num_episodes += 1
#         obs, info = env.reset()

# env.close()

# print("Finished", num_episodes, "episodes.")
# print("Average reward per episode:", np.mean(episode_rewards))
