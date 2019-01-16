import numpy as np
import matplotlib.pyplot as plt
import gym
import os

from Agent import Agent
from ParallelEnvironment import ParallelEnvironment
from keras.layers import Input, Dense, concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K

class PPOAgent(Agent):
    def __init__(self, env, epsilon=0.2, gamma=0.99, entropy_loss=1e-3, actor_lr=0.001, critic_lr=0.005,
                hidden_size=128, epochs=10, batch_size=64, buffer_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
        if type(self.env) == list:
            self.env = ParallelEnvironment(self.env)
        elif type(self.env) != ParallelEnvironment:
            self.env = ParallelEnvironment([self.env])
            
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

    def build_models(self):
        # Build Actor and Critic models
        self.models['actor'] = self.build_actor_model()
        self.models['critic'] = self.build_critic_model()       

        # When we want predictions of actions, we need to pass in three things:
        # the observation, the old prob, and the advantage.
        # Here, we just created data to spoof the last two.
        self.DUMMY_ACTION, self.DUMMY_VALUE = np.zeros((1,self.action_space.n)), np.zeros((1,1))
        
    def proximal_policy_optimization_loss(self, advantage, old_pred, debug=True):

        # Defines the PPO loss to be used during actor optimization
        def loss(y_true, y_pred):

            # advantage is a vector of size 1 passed in from the critic
            # This summing flattens it
            adv = K.sum(advantage, axis=1)
            if debug:
                adv = K.print_tensor(adv, 'advantage     :')

            # y_true is one-hot vector denoting the action taken
            # y_pred is the output of the actor neural network
            # for the given observation
            # e.g., y_true = [0,1,0], y_pred = [0.2, 0.5, 0.3]
            # so prob = y_true * y_pred = [0, 0.5, 0]
            if debug:
                y_true = K.print_tensor(y_true, 'y_true        :')
                y_pred = K.print_tensor(y_pred, 'y_pred        :')

            prob = y_true * y_pred
            if debug:
                prob = K.print_tensor(prob, 'prob          :')

            # old_pred is the actor's previous probabilty distribution
            # for the given observation
            # e.g., y_true = [0,1,0], old_pred = [0.2, 0.4, 0.4]
            # so prob = y_true * old_pred = [0, 0.4, 0]
            old_prob = y_true * old_pred
            if debug:
                old_prob = K.print_tensor(old_prob, 'old_prob      :')

            # r is the ratio of the old probability to the new one
            # e.g., r = prob/old_prob = [0, 0.5/0.4, 0]
            r = K.sum(prob/(old_prob + 1e-10), axis=1)
            if debug:
                r = K.print_tensor(r, 'r             :')

            # clipped is the value of r clipped between 1 +/- epsilon
            # e.g., r = 1.4, epsilon = 0.2 => clipped = 1.2
            clipped = K.clip(r, min_value=1-self.epsilon, max_value=1+self.epsilon)
            if debug:
                clipped = K.print_tensor(clipped, 'clipped       :')

            # minimum is the min of r * advantage and clipped * advantage
            minimum = K.minimum(r * adv, clipped * adv)
            if debug:
                minimum = K.print_tensor(minimum, 'minimum       :')

            # entropy bonus (to encourage exploration)
            entropy_bonus = self.entropy_loss * (prob * K.log(prob + 1e-10))
            entropy_bonus = K.sum(entropy_bonus, axis=1)
            if debug:
                entropy_bonus = K.print_tensor(entropy_bonus, 'entropy_bonus :')

            # K.mean computes the mean over all dimensions (left with a scaler)
            result = -K.mean(minimum + entropy_bonus)
            if debug:
                result = K.print_tensor(result, 'result        :')

            return result
        return loss

    def build_actor_model(self):
        # actor has three inputs: the current state, the advantage,
        # and the agent's predicted probability for the given observation
        state_inputs = Input(shape=self.observation_space.shape)
        advantage = Input(shape=(1,))
        old_pred = Input(shape=(self.action_space.n,))

        # hidden layers
        x = Dense(self.hidden_size, activation='relu')(state_inputs)
        x = Dense(self.hidden_size, activation='relu')(x)

        # the output is a probability distribution over the actions
        out_actions = Dense(self.action_space.n, activation='softmax')(x)

        model = Model(inputs=[state_inputs, advantage, old_pred],
                      outputs=[out_actions])

        # compile the model using our custom loss function
        model.compile(optimizer=Adam(lr=self.actor_lr),
                      loss=[self.proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_pred=old_pred
                      )])
        return model

    def build_critic_model(self):
        # critic recieves the observation as input
        state_inputs = Input(shape=self.observation_space.shape)

        # hidden layers
        x = Dense(self.hidden_size, activation='relu')(state_inputs)
        x = Dense(self.hidden_size, activation='relu')(x)

        # we predict the value of the current observation
        # i.e., cumulative discounted reward
        predictions = Dense(1, activation='linear')(x)

        model = Model(inputs=state_inputs, outputs=predictions)
        model.compile(optimizer=Adam(lr=self.critic_lr),
                      loss='mse')
        return model

    def get_action(self, observation):
        # Predict the probability destribution of the actions as a vactor
        prob = self.models['actor'].predict([np.array([observation]),
                                   self.DUMMY_VALUE,
                                   self.DUMMY_ACTION])
        prob = prob.flatten()

        # Sample an action as a scaler
        action = np.random.choice(self.action_space.n, 1, p=prob)[0]

        # Vectorize the action as a one-hot encoding
        action_vector = np.zeros(self.action_space.n)
        action_vector[action] = 1
        
        return action, action_vector, prob

    def train_batch(self, observations, actions, probabilities, rewards):
        self.logd('train_batch')
        # limit our data to the buffer_size
        obs = observations[:self.buffer_size]
        acts = actions[:self.buffer_size]
        probs = probabilities[:self.buffer_size]
        rews = rewards[:self.buffer_size]

        # our 'old probs' are really just the batches probs
        old_probs = probs

        # Calculate advantages
        values = self.models['critic'].predict(obs).reshape((self.buffer_size, 1))
        self.logd('rews.shape', rews.shape, 1)
        self.logd('values.shape', values.shape, 1)
        
        advs = rews - values
        
        self.logd('obs.shape', obs.shape, 1)
        self.logd('advs.shape', advs.shape, 1)
        self.logd('old_probs.shape', old_probs.shape, 1)
        self.logd('acts.shape', acts.shape, 1)

        # Train the actor and critic on the batch data
        self.models['actor'].fit([obs, advs, old_probs], [acts],
                       batch_size=self.batch_size, shuffle=True,
                       epochs=self.epochs, verbose=False)
        self.models['critic'].fit([obs], [advs],
                       batch_size=self.batch_size, shuffle=True,
                        epochs=self.epochs, verbose=False)
        
    def training_loop(self):
        # reset the environment
        observations = self.env.reset()
        
        # Mini batch which contains a single episode's data
        tmp_batches = []
        for env_num in range(self.env.num_envs):
            tmp_batches.append({
                'observation': [],
                'action_vector': [],
                'probability': [],
                'reward': [],
                'ep_step': 0,
                'ep_rew': []
            })       
        
        # Collect a batch of samples
        while self.episode < self.max_episodes:
            # 'Master Batch' that we add mini batches to
            batch = {
                'observation': [],
                'action_vector': [],
                'probability': [],
                'reward': []
            }

            # While we don't hit the buffer size with our master batch...
            while len(batch['observation']) < self.buffer_size:
                actions = []
                for env_num, observation in enumerate(observations):
                    tmp_batch = tmp_batches[env_num]
                    log_sub_path = 'env_{}/{}'.format(env_num, self.episode)
                    self.log_ndarray('observation', observation, tmp_batch['ep_step'], log_sub_path)
                
                    # Get the action (scalar), action vector (one-hot vector),
                    # and probability distribution (vector) from the current observation

                    action, action_vector, prob = self.get_action(observation)
                    actions.append(action)
                    
                    self.log_ndarray('action_vector', action_vector, tmp_batch['ep_step'], log_sub_path)
                    self.log_ndarray('prob', prob, tmp_batch['ep_step'], log_sub_path)

                    # Append the data to the mini batch
                    tmp_batch['observation'].append(observation)
                    tmp_batch['action_vector'].append(action_vector)
                    tmp_batch['probability'].append(prob)

                next_observations, rewards, dones, infos = self.env.step(actions)
                for env_num, (next_observation, reward, done, info) in enumerate(zip(next_observations, rewards, dones, infos)):
                    tmp_batch = tmp_batches[env_num]
                    log_sub_path = 'env_{}/{}'.format(env_num, self.episode)
                    self.log_ndarray('next_observation', next_observation, tmp_batch['ep_step'], log_sub_path)
                    self.log_ndarray('reward', reward, tmp_batch['ep_step'], log_sub_path)

                    reward_sum = np.sum(reward)
                    tmp_batch['reward'].append(reward)
                    tmp_batch['ep_rew'].append(reward_sum)
                    self.log_scalar('reward_sum', reward_sum, tmp_batch['ep_step'], log_sub_path)

                # The current observation is now the 'next' observation
                observations = next_observations

                # if the episode is at a terminal state...
                for env_num, done in enumerate(dones):
                    if not done:
                        continue
                    tmp_batch = tmp_batches[env_num]
                    log_sub_path = 'env_{}/{}'.format(env_num, self.episode)
                    episode_reward = np.sum(tmp_batch['ep_rew'])
                    self.log('Episode {} done! Reward: {} ({} steps)'.format(self.episode, episode_reward, tmp_batch['ep_step']))
                    self.log_scalar('episode_reward', episode_reward, tmp_batch['ep_step'])

                    # transform rewards based to discounted cumulative rewards
                    for j in range(len(tmp_batch['reward']) - 2, -1, -1):
                        tmp_batch['reward'][j] += tmp_batch['reward'][j + 1] * self.gamma

                    # for each entry in the mini batch...
                    for i in range(len(tmp_batch['observation'])):
                        # we unpack the data
                        obs = tmp_batch['observation'][i]
                        act = tmp_batch['action_vector'][i]
                        prob = tmp_batch['probability'][i]
                        r = tmp_batch['reward'][i]

                        # and pack it into the master batch
                        batch['observation'].append(obs)
                        batch['action_vector'].append(act)
                        batch['probability'].append(prob)
                        batch['reward'].append([r])

                    # reset the environment
                    # observations = self.env.reset()
                    
                    tmp_batches[env_num] = {
                        'observation': [],
                        'action_vector': [],
                        'probability': [],
                        'reward': [],
                        'ep_step': -1,
                        'ep_rew': []
                    }

                    # increment the episode count
                    self.episode += 1
                    self.episode_step = -1

                # END OF TRAIN STEP
                for tmp_batch in tmp_batches:
                    tmp_batch['ep_step'] += 1
                self.train_step += 1

            # we've filled up our master batch, so we unpack it into numpy arrays
            _observations = np.array(batch['observation'])
            _actions = np.array(batch['action_vector'])
            _probabilities = np.array(batch['probability'])
            _rewards = np.array(batch['reward'])

            # train the agent on the batched data
            self.train_batch(_observations, _actions, _probabilities, _rewards)