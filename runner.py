import gym
import os
import numpy as np
import torch
import datetime
from tensorboardX.writer import SummaryWriter

from algos.ppo_discrete import PPODiscrete
from common.config import Config
from common.atari_wrappers import make_atari, wrap_deepmind


class Runner:
    def __init__(self):
        self.config = Config()
        print("=====" * 10)
        print("Run on {}, train environment: {}".format(self.config.device_name, self.config.env))
        print("=====" * 10)
        self.env = make_atari(self.config.env)
        self.env = wrap_deepmind(self.env, scale=False, frame_stack=True)
        self.ppo_agent = PPODiscrete(self.config.device, self.config.obs_channels, self.config.fc_in_dim,
                                     self.config.action_dim, self.config.lr, self.config.gamma, self.config.lamb,
                                     self.config.K_epochs, self.config.upgrade_freq, self.config.eps_clip,
                                     self.config.c1, self.config.c2)

    def get_state(self, observation):
        state = np.array(observation)
        state = state.transpose((2, 0, 1))
        state = torch.FloatTensor(state)
        return state.unsqueeze(0)

    def train(self):
        observation = self.env.reset()
        state = self.get_state(observation)
        all_reward = []
        episode = 0
        episode_reward = 0
        start_time = datetime.datetime.now().replace(microsecond=0)
        step = 0
        reward_writer = SummaryWriter(self.config.reward_summary_path)
        while step < self.config.max_step:
            action = self.ppo_agent.select_action(state)
            observation, reward, done, _ = self.env.step(action)
            next_state = self.get_state(observation)

            # 存储state <s, a, r, s', done>
            self.ppo_agent.buffer.states.append(state)
            self.ppo_agent.buffer.actions.append(action)
            self.ppo_agent.buffer.rewards.append(reward)
            # self.ppo_agent.buffer.next_states.append(next_state)
            self.ppo_agent.buffer.is_terminals.append(done)

            # 更新state，episode_reward
            state = next_state
            episode_reward += reward

            # episode结束
            if done:
                reward_writer.add_scalar("reward", episode_reward, episode)
                all_reward.append(episode_reward)
                episode_reward = 0
                episode += 1
                observation = self.env.reset()
                state = self.get_state(observation)

            step += 1
            if step % self.config.upgrade_freq == 0:
                self.ppo_agent.buffer.states.append(state)
                epoch_time = datetime.datetime.now().replace(microsecond=0)
                print("step: {}, reward: {:.2f}, episode:{}, train time: {}".format(
                    step, np.mean(all_reward[-5:]), episode, epoch_time - start_time))
                self.ppo_agent.update()
                self.ppo_agent.save(self.config.ppo_model_path)
                break

    def eval(self):
        observation = self.env.reset()
        for step in range(self.config.max_step):
            action = self.env.action_space.sample()
            _, _, done, _ = self.env.step(action)
            if done:
                print(step)
                self.env.reset()
        return


