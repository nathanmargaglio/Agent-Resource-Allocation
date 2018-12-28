import numpy as np
import matplotlib.pyplot as plt
import gym

from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

from SubAgent import SubAgent

class MetaEnvironment:
    def __init__(self, agents, friction=0., seed=None):
        self.observation_spaces = []
        self.envs = []
        self.agents = agents

        for agent in self.agents:
            self.envs.append(agent.env)
            self.observation_spaces.append(agent.env.observation_space)

        self.env_count = len(self.envs)
        self.agent_count = len(self.agents)
        self.allocation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.env_count,), dtype=np.float32)
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
        self.running_reward = 0.
        self.running_rewards = [self.running_reward]

        self.uniform_episode_rewards = [0.]
        self.uniform_running_reward = 0.
        self.uniform_running_rewards = [self.uniform_running_reward]

        self.random_episode_rewards = [0.]
        self.random_running_reward = 0.
        self.random_running_rewards = [self.random_running_reward]

        self.sub_episode_rewards = [0.] * self.agent_count
        return self.observations

    def step(self, allocation):
        self.allocation = allocation
        observations = []
        rewards = []
        uniform_rewards = []
        random_rewards = []
        random_allocation = np.random.randint(self.agent_count)

        done = False
        for i, env in enumerate(self.envs):
            _action = self.agents[i].get_action(self.observations[i])[0]
            _obs, _rew, _done, _info = env.step(_action)
            self.sub_episode_rewards[i] += _rew
            rewards.append(_rew*self.allocation[i])
            uniform_rewards.append(_rew*(1./self.agent_count))
            random_rewards.append(_rew*(random_allocation == i))
            observations.append(_obs)
            done = _done

        self.observations = np.array(observations)
        cost = self.friction*abs(self.previous_allocation - self.allocation)
        rewards -= cost
        self.previous_allocation = self.allocation

        self.rewards = np.array(rewards)
        self.uniform_rewards = np.array(uniform_rewards)
        self.random_rewards = np.array(random_rewards)

        self.episode_rewards.append(np.sum(self.rewards))
        self.running_reward = np.sum(self.episode_rewards)
        self.running_rewards.append(self.running_reward)

        # Baseline Tests
        self.uniform_episode_rewards.append(np.sum(self.uniform_rewards))
        self.uniform_running_reward = np.sum(self.uniform_episode_rewards)
        self.uniform_running_rewards.append(self.uniform_running_reward)

        self.random_episode_rewards.append(np.sum(self.random_rewards))
        self.random_running_reward = np.sum(self.random_episode_rewards)
        self.random_running_rewards.append(self.random_running_reward)

        return self.observations, self.rewards, done, {}

    def render(self):
        figs = {}
        figs['sub'] = []
        for i, env in enumerate(self.envs):
            figs['sub'].append(env.render())

        fig, ax = plt.subplots(1,2,figsize=(12,4),gridspec_kw = {'width_ratios':[2, 1]})

        ax[0].set_title('MetaAgent Allocation')
        ax[0].set_ylim(0,1)
        ax[0].bar(range(self.env_count), self.allocation)

        ax[1].set_title('MetaAgent Rewards')
        ax[1].plot(self.running_rewards, label='true')
        ax[1].plot(self.uniform_running_rewards, label='uniform')
        ax[1].plot(self.random_running_rewards, label='random')
        ax[1].legend(loc='upper left')

        figs['meta'] = fig
        return figs
