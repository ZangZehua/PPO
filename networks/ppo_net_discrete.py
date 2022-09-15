import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPODiscreteNet(nn.Module):
    def __init__(self, in_channels=4, fc_in_dim=7 * 7 * 64, action_dim=5):
        super(PPODiscreteNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(fc_in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(fc_in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state):
        feature = self.conv(state)
        feature = feature.reshape(feature.shape[0], -1)
        action_probs = self.actor(feature)
        dist = Categorical(action_probs)
        state_value = self.critic(feature)
        return dist, state_value

    def act(self, state):
        feature = self.conv(state)
        feature = feature.reshape(feature.shape[0], -1)
        action_probs = self.actor(feature)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()


