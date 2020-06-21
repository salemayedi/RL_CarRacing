import torch.nn as nn
import torch
import torch.nn.functional as F


"""
CartPole network
"""

class MLP(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=400):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, action_dim)

  def forward(self, x):
    #print('this is X: ', x, type(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)


class MLP_Duel(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=400):
    super(MLP_Duel, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3_V = nn.Linear(hidden_dim, action_dim)
    self.fc3_A = nn.Linear(hidden_dim, action_dim)

  def forward(self, x):
    #print('this is X: ', x, type(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    # V
    V = self.fc3_V(x)
    # A
    Adv = self.fc3_A(x)
    # Q
    Adv_mean = torch.mean(Adv, dim = 1).unsqueeze(1)
    Q = V+ Adv - Adv_mean
    return Q


