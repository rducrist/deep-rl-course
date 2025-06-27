import torch
import numpy as np
from dqn.core import QNetwork, ReplayBuffer, train_step
import pytest

@pytest.fixture
def small_q_network():
    return QNetwork(input_dim=4, output_dim=2)

@pytest.fixture
def filled_replay_buffer():
    buffer = ReplayBuffer(capacity=10)
    for _ in range(10):
        state = np.random.rand(4)
        action = np.random.randint(2)
        reward = np.random.rand()
        next_state = np.random.rand(4)
        done = np.random.choice([0, 1])
        buffer.add((state, action, reward, next_state, done))
    return buffer

def test_train_step_updates_weights(small_q_network, filled_replay_buffer):
    q_net = small_q_network
    target_net = QNetwork(4, 2)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    old_params = [p.clone() for p in q_net.parameters()]

    train_step(q_net, target_net, filled_replay_buffer, optimizer, batch_size=5, gamma=0.99)

    new_params = list(q_net.parameters())
    assert any((not torch.equal(p0, p1)) for p0, p1 in zip(old_params, new_params)), \
        "Weights did not update during train_step"
