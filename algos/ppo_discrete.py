import math

import torch
import torch.nn as nn

from common.rollout_buffer import RolloutBuffer
# from networks.ActorCritic import ActorCritic
from networks.ppo_net_discrete import PPODiscreteNet


class PPODiscrete:
    def __init__(self, device, obs_channels, fc_in_dim, action_dim,
                 lr, gamma, lamb, K_epochs, batch_size, eps_clip, c1, c2):
        self.device = device
        self.gamma = gamma
        self.lamb = lamb
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.c1 = c1
        self.c2 = c2
        self.buffer = RolloutBuffer()

        self.policy = PPODiscreteNet(obs_channels, fc_in_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPODiscreteNet(obs_channels, fc_in_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = state.to(self.device)
            action = self.policy_old.act(state)
            return action

    def update(self):
        # get all data from the buffer
        state_batch = torch.cat(self.buffer.states[:-1]).to(self.device)
        action_batch = torch.Tensor(self.buffer.actions).to(self.device)  # has no grad, no need to use detach
        reward_batch = torch.Tensor(self.buffer.rewards).to(self.device)  # has no grad, no need to use detach
        next_state_batch = torch.cat(self.buffer.states[1:]).to(self.device)
        done_batch = torch.Tensor(self.buffer.is_terminals).to(self.device)

        # compute state value and dist
        dists, states_value = self.policy(state_batch)
        _, next_states_value = self.policy(next_state_batch)

        # compute advantage
        advantage_batch = []
        if done_batch[self.batch_size - 1]:
            next_states_value[self.batch_size - 1] = 0
        delta = reward_batch[self.batch_size - 1] + self.gamma * next_states_value[self.batch_size - 1] - states_value[
            self.batch_size - 1]
        advantage = delta
        advantage_batch.append(advantage)
        for t in range(state_batch.shape[0] - 2, -1, -1):
            if done_batch[t]:  # True
                next_states_value[t] = 0
            delta = reward_batch[t] + self.gamma * next_states_value[t] - states_value[t]
            advantage = delta + self.gamma * self.lamb * advantage
            advantage_batch.append(advantage)
        advantage_batch.reverse()
        advantage_batch = torch.Tensor(advantage_batch).to(self.device)

        # train K_epochs times using the same data
        for epoch in range(self.K_epochs):
            # compute state value and dist
            dists, states_value = self.policy(state_batch)
            _, next_states_value = self.policy(next_state_batch)
            dists_old, _ = self.policy_old(state_batch)

            # Loss CLIP part
            # ration pi/pi_old = exp(log(pi/pi_old)) = exp(log(pi)-log(pi_old))
            pi = dists.log_prob(action_batch)
            pi_old = dists_old.log_prob(action_batch).detach()
            ration = torch.exp(pi - pi_old)
            clipped_ration = torch.clamp(ration, 1 - self.eps_clip, 1 + self.eps_clip)

            unclipped_l = ration * advantage_batch
            clipped_l = clipped_ration * advantage_batch

            l_CLIP = torch.mean(torch.min(unclipped_l, clipped_l))

            # loss VF part
            # clip state value
            loss_func = nn.MSELoss()
            part = reward_batch + self.gamma * next_states_value
            clipped_value = torch.clamp(states_value, 1 - self.eps_clip, 1 + self.eps_clip)
            unclipped_l = loss_func(part, states_value)
            clipped_l = loss_func(part, clipped_value)
            l_VF = torch.mean(torch.max(unclipped_l, clipped_l))

            # loss S entropy bonus
            l_S = torch.mean(dists.entropy())

            loss = l_CLIP + self.c1 * l_VF + self.c2 * l_S
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(self.policy.state_dict())
        print(self.policy_old.state_dict())
        # upgrade old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
