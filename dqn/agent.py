import torch
import random
import numpy as np
import torch.nn.functional as F
from .model import QNetwork
from .storage import ReplayBuffer


class Agent():
    """Implementation of a DQN Agent
    """

    def __init__(self, state_size, action_size, device,
                 buffer_size=int(1e5), batch_size=64,
                 gamma=0.99, tau=1e-3, seed=1, lr=5e-4, update_every=4):
        """Initialize a DQN Agent
        Params
        ======
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            device: cpu or gpu to use
            buffer_size (int): size of the replay buffer
            batch_size (int): size of the batch to sample from replay buffer
            gamma (float): discount factor
            tau (float): factor for soft update
            seed (int): random seed
            lr (float): learning rate of the network
            update_every (int): steps after which the target network will be updated.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.seed = random.seed(seed)
        self.update_every = update_every
        self.qnetwork_local = QNetwork(
            state_size=state_size, action_size=action_size, seed=seed).to(self.device)
        self.qnetwork_target = QNetwork(
            state_size=state_size, action_size=action_size, seed=seed).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.qnetwork_local.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size, seed)
        self.time_step = 0

    def learn(self, experiences):
        """Helper function to optimize the local network from a single batch experience tuples
        Params
        ======
            experiences: a batch of experince tuple
        """
        states, actions, rewards, next_states, dones = experiences
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

    def soft_update(self):
        """Helper function to update the target netork.
        """
        for local_param, target_param in zip(self.qnetwork_local.parameters(), self.qnetwork_target.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def step(self, state, action, reward, next_state, done):
        """Store the experience in memory and learn 
        from a batch of experience sampled from the replay buffer.

        Params
        ======
            state: current state
            action: action taken by the agent
            reward: reward received by taking the action
            next_state: next state
            done: if the agent won or the game is over 
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.time_step = (self.time_step + 1) % self.update_every
        if self.time_step == 0:
            if len(self.replay_buffer) >= self.batch_size:
                experiences = self.replay_buffer.sample(self.batch_size, self.device)
                self.learn(experiences)

    def act(self, state, epsilon=0.0):
        """Select an action based on the current policy
        Params
        ======
            state: state for which to select action
            epsilon (float): factor for epsilon greedy method
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
