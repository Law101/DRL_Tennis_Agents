# <------------- This file contains implementation of the Neural Network) ---------------------->

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


# <----------------- Define the Actor Class ---------------------->
class Actor(nn.Module):
    """ The Actor (Policy) Model """
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=150):
        """
        Parameters
        ----------------
        state_size(int): Dimension of each state
        action_size(int): dimension of each action
        seed (int): Random seed
        fc1_units (int): Number of Nodes in the first hidden layer
        fc2_units (int): Number of Nodes in the second hidden layer
        """
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size*2, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()
        
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            
            
    #Map States to corresponding Actions    
    def forward(self, state):
        """
        An Actor (Policy) network that maps states to actions.
        Activation Function: ReLU and tanh
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
    
    
    
# <----------------- Define the Critic Class ---------------------->
class Critic(nn.Module):
    """ Critic (Value) Model """
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=150):
        
        """
        Parameters
        ----------------
        state_size(int): Dimension of each state
        action_size(int): dimension of each action
        seed (int): Random seed
        fc1_units (int): Number of Nodes in the first hidden layer
        fc2_units (int): Number of Nodes in the second hidden layer
        """
        
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size*2, fc1_units)
        self.fc2 = nn.Linear(fc1_units + (action_size*2), fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
        
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        
     # Map (state, action) pair ----> Q_values
    def forward(self, state, action):
        """
        Build a critic (value) network that maps (state, action) pairs --> Q_Values.
        Activation Function: ReLU
        """
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)