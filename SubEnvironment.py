import numpy as np
import matplotlib.pyplot as plt
import gym

from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

class SubEnvironment:
    def __init__(self, friction=None, lookback=10, max_steps=100,
            noise=None, amplitude=None, frequency=None, phase=None, seed=None):
        # cost of changing position
        self.fric = self.handle_param(friction, 0.1)

        # how much of the value history the agent sees
        self.lookback = lookback

        # available actions (hold/invest)
        self.action_space = gym.spaces.Discrete(2)

        # how long an episode lasts
        self.max_steps = max_steps

        # noise to apply to the value
        self.noise_param = noise

        # max amplitude of value
        self.amp_param = amplitude

        # frequency of value
        self.freq_param = frequency

        # phase (t-offset) of value
        self.phase_param = phase

        # observation space (position + lookback)
        self.observation_space = gym.spaces.Box(low=-1., high=1.0,
                shape=(self.lookback+1,), dtype=np.float32)

        # set a seed for testing
        self.seed = seed

    def handle_param(self, param, default=None):
        # frequency of value
        if param is None and default is not None:
            # default
            return default
        elif type(param) == tuple:
            # if a tuple is passed, a random value is picked in the range
            return (param[1] - param[0])*np.random.rand() + param[1]
        else:
            # we assume it's a number, and set it directly
            return float(param)


    def reset(self, seed=None):
        if seed is None:
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
        self.noise = self.handle_param(self.noise_param, 0)
        self.amp = self.handle_param(self.amp_param, 1.)
        self.freq = self.handle_param(self.freq_param, 4)
        self.phase = self.handle_param(self.phase_param, 0)*self.max_steps

        self.a_space = (1-self.noise)*self.amp*np.sin(np.linspace(self.phase,self.freq*np.pi+self.phase,self.max_steps))
        self.a_space += np.random.normal(0, self.amp*self.noise, size=self.a_space.shape)

        # The current position of the agent (0: neutral, 1: long)
        self.position = 0

        # The current position (as a singleton) concatenated with
        # the current "viewable" observation
        self.observation = np.concatenate([[self.position], self.a_space[self.t-self.lookback:self.t]])

        return self.observation.reshape(self.observation_space.shape)

    def step(self, action):
        self.instantaneous_reward = 0
        # if we changed position...
        if self.position != action:
            # then we incur transaction costs
            self.instantaneous_reward -= self.fric*self.amp

            # and change our position
            self.position = action

        # increment time
        self.t += 1

        # update observation
        self.observation = np.concatenate([[self.position], self.a_space[self.t-self.lookback:self.t]])

        # our current asset value
        current = self.observation[-1]

        # the reward for our position
        # e.g., if our position == 0 (neutral), then we gain no reward
        # if our position == 1 (long), then we gain/lose the change
        self.instantaneous_reward += self.position * current
        self.current_reward += self.instantaneous_reward
        self.episode_rewards.append(self.current_reward)

        done = False
        if self.t >= self.max_steps - 1:
            done = True

        return self.observation.reshape(self.observation_space.shape), self.instantaneous_reward, done, {}

    def render(self):
        fig, ax = plt.subplots(1,2,figsize=(12,4),gridspec_kw = {'width_ratios':[2, 1]})

        ax[0].plot(self.x_space, self.a_space)
        ax[0].plot([self.t], [self.a_space[self.t]], 'o')
        ax[1].plot(self.episode_rewards)

        plt.show()
