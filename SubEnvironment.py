import numpy as np
import matplotlib.pyplot as plt
import gym

from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

class SubEnvironment:
    def __init__(self, lookback=10, max_steps=100, noise=0, seed=None):
        # how much of the value history the agent sees
        self.lookback = lookback

        # available actions (hold/invest)
        self.action_space = gym.spaces.Discrete(2)

        # observation space (lookback)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.lookback,), dtype=np.float32)

        # how long an episode lasts
        self.max_steps = max_steps

        # noise to apply to the value
        self.noise = noise

        # set a seed for testing
        self.seed = seed

    def reset(self, seed=None):
        if self.seed:
            np.random.seed(self.seed)
        if seed:
            np.random.seed(seed)

        # The current timestep
        self.t = self.lookback

        # The rewards of the episode
        self.current_reward = 0.
        self.episode_rewards = [self.current_reward]

        # Our "time" space
        self.x_space = np.arange(self.max_steps)

        # Our "asset" space
        a_c = np.random.randint(2,8)
        self.a_space = (1-self.noise)*np.sin(np.linspace(0,a_c*np.pi,self.max_steps))
        self.a_space += np.random.normal(0, self.noise, size=self.a_space.shape)
        self.a_space = np.clip(self.a_space, -1., 1.)

        # The current "viewable" observation
        self.observation = self.a_space[self.t-self.lookback:self.t]

        # The current position of the agent (0: neutral, 1: long)
        self.position = 0

        return self.observation.reshape(self.observation_space.shape)

    def step(self, action):
        # increment time
        self.t += 1

        # update observation
        self.observation = self.a_space[self.t-self.lookback:self.t]

        # our current asset value
        current = self.observation[-1]

        # our current position
        self.position = action

        # the reward for our position
        # e.g., if our position == 0 (neutral), then we gain no reward
        # if our position == 1 (long), then we gain/lose the change
        instantaneous_reward = self.position * current
        self.current_reward += instantaneous_reward
        self.episode_rewards.append(self.current_reward)

        done = False
        if self.t >= self.max_steps - 1:
            done = True

        return self.observation.reshape(self.observation_space.shape), instantaneous_reward, done, {}

    def render(self):
        fig, ax = plt.subplots(1,2,figsize=(12,4),gridspec_kw = {'width_ratios':[2, 1]})

        ax[0].plot(self.x_space, self.a_space)
        ax[0].plot([self.t], [self.a_space[self.t]], 'o')
        ax[1].plot(self.episode_rewards)

        plt.show()
