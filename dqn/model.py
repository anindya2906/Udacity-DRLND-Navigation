import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Q Learning Model
    """

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, seed=1):
        """Initialize a QNetwork model
        Params
        ======
            state_size (int): size of the state space
            action_size (int): size of the action space
            fc1_units (int): size of the first hidden layer
            fc2_units (int): size of the seconf hidden layer
            seed (int): random seed 
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """The forward pass of the neural network
        Params
        ======
            state: the state for which the Q values of the action need to predicted
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output
