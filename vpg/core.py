import torch 
import gymnasium as gym 
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

obs, info = env.reset(seed=42)