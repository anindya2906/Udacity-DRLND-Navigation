import torch
import torch.nn as nn
import torch.nn.Functional as F


class QNetwork(nn.Model):

    def __init__(self, state_size, action_size, fc1_size=64, fc2_size=64, seed=1):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output
