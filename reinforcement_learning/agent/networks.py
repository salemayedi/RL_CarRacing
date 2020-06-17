import torch.nn as nn
import torch
import torch.nn.functional as F


"""
CartPole network
"""

class MLP(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=32):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim*2)
    self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, action_dim)

  def forward(self, x):
    #print('this is X: ', x, type(x))
    x = F.tanh(self.fc1(x))
    x = F.tanh(self.fc2(x))
    return self.fc3(x)


