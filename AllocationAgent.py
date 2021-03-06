import numpy as np
import matplotlib.pyplot as plt
import gym

from Agent import Agent
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

class AllocationAgent(Agent):
    def __init__(self, env, epsilon=0.05, gamma=0.99, entropy_loss=1e-3, actor_lr=1e-4, critic_lr=1e-4,
                 hidden_size=128, epochs=8, batch_size=64, buffer_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.env = env
        
        # These properties require a custom environment
        self.envs = env.envs
        self.env_count = env.env_count

        self.agents = env.agents
        self.agent_count = env.agent_count

        self.allocation_space = env.allocation_space
        self.observation_spaces = env.observation_spaces

        # Set hyperparameters
        self.epsilon = epsilon
        self.gamma = gamma
        self.entropy_loss = entropy_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Build Actor and Critic models
        self.models['actor'] = self.build_actor_model()
        self.models['critic'] = self.build_critic_model()
        
        self.DUMMY_ALLOCATION = np.zeros(self.allocation_space.shape)
        self.DUMMY_VALUE = np.zeros((1, self.agent_count))

    def allocation_proximal_policy_optimization_loss(self, advantage, previous_allocation, debug=True):
        def loss(y_true, y_pred):
            adv = advantage
            if debug:
                adv = K.print_tensor(adv, 'advantage     :')

            if debug:
                y_true = K.print_tensor(y_true, 'y_true        :')
                y_pred = K.print_tensor(y_pred, 'y_pred        :')

            alloc = y_true * y_pred
            if debug:
                alloc = K.print_tensor(alloc, 'alloc          :')

            prev_alloc = y_true * previous_allocation
            if debug:
                prev_alloc = K.print_tensor(prev_alloc, 'prev_alloc      :')

            r = alloc/(prev_alloc + 1e-10)
            if debug:
                r = K.print_tensor(r, 'r             :')

            clipped = K.clip(r, min_value=1-self.epsilon, max_value=1+self.epsilon)
            if debug:
                clipped = K.print_tensor(clipped, 'clipped       :')

            minimum = K.minimum(r * adv, clipped * adv)
            if debug:
                minimum = K.print_tensor(minimum, 'minimum       :')

            entropy_bonus = -self.entropy_loss * (alloc * K.log(alloc + 1e-10))
            if debug:
                entropy_bonus = K.print_tensor(entropy_bonus, 'entropy_bonus :')

            result = -K.mean(minimum + entropy_bonus)
            if debug:
                result = K.print_tensor(result, 'result        :')

            return result
        return loss

    def build_actor_model(self):
        observation_inputs = []
        for i, sub_obs in enumerate(self.observation_spaces):
            sub_input = Input(shape=sub_obs.shape, name='sub_obs_{}'.format(i))
            observation_inputs.append(sub_input)
        advantage = Input(shape=(self.agent_count,), name='advantage')
        previous_allocation = Input(shape=self.allocation_space.shape, name='previous_allocation')

        x = concatenate(observation_inputs + [previous_allocation])
        x = Dense(self.hidden_size, activation='relu')(x)
        x = Dense(self.hidden_size, activation='relu')(x)

        out_allocation = Dense(self.allocation_space.shape[0], activation='softmax')(x)

        model = Model(inputs=observation_inputs + [advantage, previous_allocation],
                      outputs=[out_allocation])

        model.compile(optimizer=Adam(lr=self.actor_lr),
                      loss=[self.allocation_proximal_policy_optimization_loss(
                          advantage=advantage,
                          previous_allocation=previous_allocation
                      )])
        return model

    def build_critic_model(self):
        # critic recieves the observation as input
        observation_inputs = []
        for i, sub_obs in enumerate(self.observation_spaces):
            sub_input = Input(shape=sub_obs.shape, name='sub_obs_{}'.format(i))
            observation_inputs.append(sub_input)
        previous_allocation = Input(shape=self.allocation_space.shape, name='previous_allocation')

        # hidden layers
        x = concatenate(observation_inputs + [previous_allocation])
        x = Dense(self.hidden_size, activation='relu')(x)
        x = Dense(self.hidden_size, activation='relu')(x)

        # we predict the value of the current observation
        # i.e., cumulative discounted reward
        values = Dense(self.allocation_space.shape[0], activation='linear')(x)

        model = Model(inputs=observation_inputs + [previous_allocation],
                      outputs=[values])
        model.compile(optimizer=Adam(lr=self.critic_lr), loss='mse', metrics=['mse'])
        return model

    def get_allocation(self, observations, prev_alloc):
        p_al = np.array([prev_alloc])
        obs = [o.reshape((1,) + o.shape) for o in observations]
        alloc = self.models['actor'].predict(obs + [self.DUMMY_VALUE, p_al])
        return alloc

    def train_batch(self, observations, allocations, rewards, previous_allocations):
        # limit our data to the buffer_size
        obs = observations[:self.buffer_size]
        allocs = allocations[:self.buffer_size]
        rews = rewards[:self.buffer_size]

        prev_allocs = previous_allocations[:self.buffer_size]
        obs = np.split(obs, obs.shape[1], axis=1)
        obs = [o.reshape(o.shape[0], o.shape[2]) for o in obs]

        values = self.models['critic'].predict(obs + [prev_allocs])
        advs = rews - values

        self.models['actor'].fit(obs + [advs, prev_allocs], [allocs],
                       batch_size=self.batch_size, shuffle=True,
                       epochs=self.epochs, verbose=False)
        self.models['critic'].fit(obs + [prev_allocs], [advs],
                       batch_size=self.batch_size, shuffle=True,
                       epochs=self.epochs, verbose=False)

    def run(self, episodes, update_version=True):
        self.episode = 0
        self.train_step = 0
        self.episode_step = 0
        episode_rewards = []
        
        self.create_new_run(update_version)

        self.log("Starting run: {} v{}".format(self.name, self.version))
        
        # reset the environment
        observations = self.env.reset()

        # Beginning of Train Step
        while self.episode < episodes:
            # 'Master Batch' that we add mini batches to
            batch = {
                'observation': [],
                'allocation_vector': [],
                'previous_allocation_vector': [],
                'reward': []
            }

            # Mini batch which contains a single episode's data
            tmp_batch = {
                'observation': [],
                'allocation_vector': [],
                'previous_allocation_vector': [],
                'reward': []
            }

            # The Allocation Agent needs the "previous" allocation
            previous_alloc_vector = self.env.allocation
            tmp_batch['previous_allocation_vector'].append(previous_alloc_vector)
            
            # BEGINNING OF TRAIN STEP
            # While we don't hit the buffer size with our master batch...
            while len(batch['observation']) < self.buffer_size:
                self.log_ndarray('observations', observations, self.episode_step, self.episode)
                self.log_ndarray('previous_alloc_vector', previous_alloc_vector, self.episode_step, self.episode)
                
                # Get the action (scalar), action vector (one-hot vector),
                # and probability distribution (vector) from the current observation
                alloc_vector = self.get_allocation(observations, previous_alloc_vector)[0]
                self.log_ndarray('alloc_vector', alloc_vector, self.episode_step, self.episode)
                
                # Get the next observation, reward, done, and info for taking an action
                next_observations, rewards, done, info = self.env.step(alloc_vector)

                self.log_ndarray('next_observations', next_observations, self.episode_step, self.episode)
                self.log_ndarray('rewards', rewards, self.episode_step, self.episode)
                
                reward_sum = np.sum(rewards)
                episode_rewards.append(reward_sum)
                self.log_scalar('reward_sum', reward_sum, self.episode_step, self.episode)
                
                # Append the data to the mini batch
                tmp_batch['observation'].append(observations)
                tmp_batch['allocation_vector'].append(alloc_vector)
                tmp_batch['previous_allocation_vector'].append(previous_alloc_vector)
                tmp_batch['reward'].append(rewards)

                # The current observation is now the 'next' observation
                observations = next_observations
                previous_alloc_vector = alloc_vector

                # if the episode is at a terminal state...
                if done:
                    episode_reward = np.sum(episode_rewards)
                    self.log_scalar('episode_reward', episode_reward, self.episode_step, self.episode)

                    # transform rewards based to discounted cumulative rewards
                    for j in range(len(tmp_batch['reward']) - 2, -1, -1):
                        tmp_batch['reward'][j] += tmp_batch['reward'][j + 1] * self.gamma

                    # for each entry in the mini batch...
                    for i in range(len(tmp_batch['observation'])):
                        # we unpack the data
                        obs = tmp_batch['observation'][i]
                        alloc = tmp_batch['allocation_vector'][i]
                        previous_alloc = tmp_batch['previous_allocation_vector'][i]
                        r = tmp_batch['reward'][i]

                        # and pack it into the master batch
                        batch['observation'].append(obs)
                        batch['allocation_vector'].append(alloc)
                        batch['previous_allocation_vector'].append(previous_alloc)
                        batch['reward'].append(r)

                    #self.log_scalar('reward', self.env.running_reward, self.episode, 'meta')
                    #self.log_scalar('reward', self.env.uniform_running_reward, self.episode, 'uni')
                    #self.log_scalar('reward', self.env.random_running_reward, self.episode, 'rand')

                    # reset the environment
                    observations = self.env.reset()

                    # reset the mini batch
                    tmp_batch = {
                        'observation': [],
                        'allocation_vector': [],
                        'previous_allocation_vector': [],
                        'reward': []
                    }

                    # increment the episode count
                    self.episode += 1
                    self.episode_step = -1
                    episode_rewards = []

                # END OF TRAIN STEP
                self.episode_step += 1
                self.train_step += 1

            # we've filled up our master batch, so we unpack it into numpy arrays
            _observations = np.array(batch['observation'])
            _allocs = np.array(batch['allocation_vector'])
            _prev_allocs = np.array(batch['previous_allocation_vector'])
            _rewards = np.array(batch['reward'])

            # train the agent on the batched data
            self.train_batch(_observations, _allocs, _rewards, _prev_allocs)