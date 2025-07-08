import torch
import gymnasium as gym
from collections import deque
from torch import nn
import numpy as np

env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
action_high = env.action_space.high
action_low = env.action_space.low


