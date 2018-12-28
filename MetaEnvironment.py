import numpy as np
import matplotlib.pyplot as plt
import gym

from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

from SubAgent import SubAgent

class MetaEnvironment:
    def __init__(self, envs, friction=0., seed=None):
        self.envs = envs
        self.env_count = len(envs)

        self.allocation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.env_count,), dtype=np.float32)
        self.observation_spaces = []
        self.agents = []
        for env in self.envs:
            self.agents.append(SubAgent(env))
            self.observation_spaces.append(env.observation_space)

        self.agent_count = len(self.agents)
        self.friction = friction
        self.seed = seed

    def reset(self):
        self.allocation = np.ones(shape=(self.env_count,))
        self.allocation /= np.sum(self.allocation)
        self.previous_allocation = self.allocation

        observations = []
        rewards = []
        for env in self.envs:
            obs = env.reset()
            rewards.append(0)
            observations.append(obs)

        self.observations = np.array(observations)
        self.rewards = np.array(rewards)
        self.episode_rewards = [0.]
        self.running_rewards = [0.]
        return self.observations

    def step(self, allocation):
        self.allocation = allocation
        observations = []
        rewards = []

        done = False
        for i, env in enumerate(self.envs):
            _action = self.agents[i].get_action(self.observations[i])[0]
            _obs, _rew, _done, _info = env.step(_action)
            rewards.append(_rew)
            observations.append(_obs)
            done = _done

        self.observations = np.array(observations)
        cost = self.friction*abs(self.previous_allocation - self.allocation)
        rewards -= cost
        self.rewards = np.array(rewards)
        self.previous_allocation = self.allocation

        self.episode_rewards.append(np.sum(self.rewards))
        self.running_rewards.append(np.sum(self.episode_rewards))
        return self.observations, self.rewards, done, {}

    def render(self):
        figs = {}
        figs['sub'] = []
        for i, env in enumerate(self.envs):
            figs['sub'].append(env.render())

        fig, ax = plt.subplots(1,2,figsize=(12,4),gridspec_kw = {'width_ratios':[2, 1]})

        ax[0].set_ylim(0,1)
        ax[0].bar(range(self.env_count), self.allocation)
        ax[1].plot(self.running_rewards)

        figs['meta'] = fig
        return figs
