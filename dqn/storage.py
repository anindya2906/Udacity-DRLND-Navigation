import random
import torch
import numpy as np
from collections import deque, namedtuple


class ReplayBuffer():
    """Helper class to store and sample experience tuples.
    """

    def __init__(self, buffer_size, seed):
        """Initialize a replay buffer object
        Params
        ======
            buffer_size (int): size of the buffer
            seed (int): random seed
        """
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        """Store a single experience tuple in memory
        Params
        ======
            state: current state
            action: action taken by the agent
            reward: reward received by taking the action
            next_state: next state
            done: if the agent won or the game is over 
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size, device):
        """Sample a random batch from the memory
        Params
        ======
            batch_size (int): batch size of the memory
            device: cpu or gpu
        """
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
