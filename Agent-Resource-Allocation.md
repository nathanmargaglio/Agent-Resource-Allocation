

```python
# import jtplot submodule from jupyterthemes
from jupyterthemes import jtplot

# currently installed theme will be used to
# set plot style if no arguments provided
jtplot.style()
```


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym

from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
```

## SubEnvironment

This environment models a single asset that oscillates via Sine wave with optional noise.  The asset fluctuates in value, and the agent must either hold (not do anything) or invest in the asset.

The reward is based on the agent's decision and the value of the asset.  If the decision is to hold, then the reward is 0.  If the decision is to invest, then the reward is equal to the value.  The value can range from -1 to 1.

Ideally, the agent will properly forecast the next timestep's value, and choose to hold if it's negative and to invest if it's positive.


```python
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
```

Next, we test the subenvironment.  Here, we take random actions after every step and plot every 10 steps.


```python
env = SubEnvironment(noise=0.5)
env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    if (env.t + 1) % 25 == 0:
        env.render()
```


![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_5_0.png)



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_5_1.png)



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_5_2.png)



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_5_3.png)


#### SubAgent

Here, we define the sub-agent.  This is simply a PPO agent with no modifications.


```python
class SubAgent:
    def __init__(self, env,
                epsilon=0.2, gamma=0.99, entropy_loss=1e-3, actor_lr=0.001, critic_lr=0.005,
                hidden_size=128, epochs=10, batch_size=64, buffer_size=256):
        # Clear Tensorflow session and set some metadata
        # K.clear_session()
        self.env = env
        
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
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
        self.actor = self.build_actor() 
        self.critic = self.build_critic()
        
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
    
    def build_actor(self):
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
    
    def build_critic(self):
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
        prob = self.actor.predict([np.array([observation]), 
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
        # limit our data to the buffer_size
        obs = observations[:self.buffer_size]
        acts = actions[:self.buffer_size]
        probs = probabilities[:self.buffer_size]
        rews = rewards[:self.buffer_size]
        
        # our 'old probs' are really just the batches probs
        old_probs = probs
        
        # Calculate advantages
        values = self.critic.predict(obs).reshape((self.buffer_size, 1))
        advs = rews - values
            
        # Train the actor and critic on the batch data
        self.actor.fit([obs, advs, old_probs], [acts],
                       batch_size=self.batch_size, shuffle=True,
                       epochs=self.epochs, verbose=False)
        self.critic.fit([obs], [rews],
                       batch_size=self.batch_size, shuffle=True,
                        epochs=self.epochs, verbose=False)
    
    def run(self, episodes, verbose=False):
        episode = 0
        reward_history = []

        # reset the environment
        observation = self.env.reset()

        # Collect a batch of samples
        while episode < episodes:
            # 'Master Batch' that we add mini batches to
            batch = {
                'observation': [],
                'action_vector': [],
                'probability': [],
                'reward': []
            }

            # Mini batch which contains a single episode's data
            tmp_batch = {
                'observation': [],
                'action_vector': [],
                'probability': [],
                'reward': []
            }

            # While we don't hit the buffer size with our master batch...
            while len(batch['observation']) < self.buffer_size:
                # Get the action (scalar), action vector (one-hot vector), 
                # and probability distribution (vector) from the current observation
                action, action_vector, prob = self.get_action(observation)

                # Get the next observation, reward, done, and info for taking an action
                next_observation, reward, done, info = self.env.step(action)

                # Append the data to the mini batch
                tmp_batch['observation'].append(observation)
                tmp_batch['action_vector'].append(action_vector)
                tmp_batch['probability'].append(prob)
                tmp_batch['reward'].append(reward)

                # The current observation is now the 'next' observation
                observation = next_observation

                # if the episode is at a terminal state...
                if done:
                    # log some reward data (for plotting)
                    reward_data = np.sum(tmp_batch['reward'])
                    reward_history.append(reward_data)

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
                        batch['reward'].append(r)

                    # every 10th episode, log some stuff
                    if (episode+1) % 10 == 0 and verbose:
                        print('Episode:', episode)
                        print('Reward :', reward_data)
                        print('Average:', np.mean(reward_history[-100:]))
                        print('-'*10)
                        print()

                        self.env.render()
                
                    # reset the environment
                    observation = self.env.reset()

                    # reset the mini batch
                    tmp_batch = {
                        'observation': [],
                        'action_vector': [],
                        'probability': [],
                        'reward': []
                    }

                    # increment the episode count
                    episode += 1

            # we've filled up our master batch, so we unpack it into numpy arrays
            observations = np.array(batch['observation'])
            actions = np.array(batch['action_vector'])
            probabilities = np.array(batch['probability'])
            rewards = np.array(batch['reward'])    
            rewards = np.reshape(rewards, (len(batch['reward']), 1))

            # train the agent on the batched data
            self.train_batch(observations, actions, probabilities, rewards)
                    
        self.reward_history = reward_history
        return self.reward_history
```

## Training SubAgents

Here, we demonstrate how the subagents can be trained to "trade" the assets so as to maximize "profit."


```python
env = SubEnvironment(noise=0.3)
agent = SubAgent(env=env)
rh = agent.run(100, verbose=True)
```

    Episode: 9
    Reward : -0.12889498999117122
    Average: 2.2958844470369004
    ----------
    



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_9_1.png)


    Episode: 19
    Reward : 14.07373483472441
    Average: 4.816181604886628
    ----------
    



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_9_3.png)


    Episode: 29
    Reward : 15.334031913324164
    Average: 6.082777586793421
    ----------
    



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_9_5.png)


    Episode: 39
    Reward : 5.648983741084642
    Average: 6.029916711495092
    ----------
    



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_9_7.png)


    Episode: 49
    Reward : 8.148431789490884
    Average: 5.845277195847193
    ----------
    



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_9_9.png)


    Episode: 59
    Reward : 7.706493883461285
    Average: 5.998233630300222
    ----------
    



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_9_11.png)


    Episode: 69
    Reward : 22.913036401164383
    Average: 6.924747063998773
    ----------
    



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_9_13.png)


    Episode: 79
    Reward : 6.855803862398463
    Average: 7.5881861623239075
    ----------
    



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_9_15.png)


    Episode: 89
    Reward : 15.73754014467619
    Average: 8.232603487162873
    ----------
    



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_9_17.png)


    Episode: 99
    Reward : 13.902649437202596
    Average: 8.902823715111836
    ----------
    



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_9_19.png)



```python
rh_mean = pd.Series(rh).rolling(50).mean()
fig, ax = plt.subplots(1,1,figsize=(8,4))

ax.set_title('Training Episode Rewards')
ax.set_xlabel('Episode')
ax.set_ylabel('Average Reward (previous 50)')
ax.plot(rh_mean)

plt.show()
```


![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_10_0.png)


Here, we test our trained subagent on a seeded environment.


```python
env = SubEnvironment(seed=1, noise=0.3)
obs = env.reset()
done = False
reward = 0
while not done:
    action = agent.get_action(obs)[0]
    obs, rew, done, info = env.step(action)
    reward += rew
    if (env.t + 1) % 25 == 0:
        env.render()
        
print('Reward:', reward)
```


![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_12_0.png)



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_12_1.png)



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_12_2.png)



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_12_3.png)


    Reward: 12.955920194312117


Then, for comparison, we test the same environment as above on an untrained agent.


```python
env = SubEnvironment(seed=1, noise=0.3)
untrained_agent = SubAgent(env=env)
obs = env.reset()
done = False
reward = 0
while not done:
    action = untrained_agent.get_action(obs)[0]
    obs, rew, done, info = env.step(action)
    reward += rew
    if (env.t + 1) % 25 == 0:
        env.render()
        
print('Reward:', reward)
```


![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_14_0.png)



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_14_1.png)



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_14_2.png)



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_14_3.png)


    Reward: -3.854374519368652


As we can see, the trained agent performs considerably better than the untrained agent.  Granted, the task is relatively easy, but the main point is that our subagents can perform better on the environment than a random agent.

## MetaEnvironment

Next, we create the MetaEnvironment, which acts as a higher level environment for organizing a collect of subenvironments.


```python
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
        for i, env in enumerate(self.envs):
            print('Sub {}'.format(i))
            env.render()
            
        fig, ax = plt.subplots(1,2,figsize=(12,4),gridspec_kw = {'width_ratios':[2, 1]})

        ax[0].set_ylim(0,1)
        ax[0].bar(range(self.env_count), self.allocation)
        ax[1].plot(self.running_rewards)

        print("Meta")
        plt.show()
```

Test MetaEnvironment


```python
sub_envs = []
for i in range(3):
    sub_envs.append(SubEnvironment(noise=0.3))
    
meta_env = MetaEnvironment(sub_envs)
obs = meta_env.reset()

obs = env.reset()
done = False
i = 0
while not done:
    allocation = meta_env.allocation_space.sample()
    obs, rew, done, info = meta_env.step(allocation)
    i += 1
    if (i + 1) % 25 == 0:
        print()
        meta_env.render()
```

    
    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_1.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_3.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_5.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_7.png)


    
    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_9.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_11.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_13.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_15.png)


    
    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_17.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_19.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_21.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_20_23.png)


## MetaAgent


```python
class MetaAgent:
    def __init__(self, env,
                epsilon=0.2, gamma=0.99, entropy_loss=1e-2, actor_lr=0.001, critic_lr=0.005,
                hidden_size=128, epochs=10, batch_size=64, buffer_size=256, seed=None):
        self.env = env
        
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
        self.coef = 0.01
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Build Actor and Critic models
        self.actor = self.build_actor() 

        self.DUMMY_ALLOCATION = np.zeros(self.allocation_space.shape)
        self.DUMMY_VALUE = np.zeros((1, self.agent_count))

    def proximal_policy_optimization_loss(self, advantage, debug=True):
        def loss(y_true, y_pred):
            y_pred = K.print_tensor(y_pred, 'pred ')
            adv = K.print_tensor(advantage, 'adva ')
            return -self.coef*adv*y_pred + self.entropy_loss*y_pred*K.log(y_pred+1e-10)
        return loss
    
    def build_actor(self):
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
                      loss=[self.proximal_policy_optimization_loss(
                          advantage=advantage
                      )])
        return model

    def get_allocation(self, observations, prev_alloc):
        p_al = np.array([prev_alloc])
        obs = [o.reshape((1,) + o.shape) for o in observations]
        alloc = self.actor.predict(obs + [self.DUMMY_VALUE, p_al])
        return alloc
    
    def train_batch(self, observations, allocations, rewards, previous_allocations):
        # limit our data to the buffer_size
        obs = observations[:self.buffer_size]
        allocs = allocations[:self.buffer_size]
        rews = rewards[:self.buffer_size]
        prev_allocs = previous_allocations[:self.buffer_size]
        obs = np.split(obs, obs.shape[1], axis=1)
        obs = [o.reshape(o.shape[0], o.shape[2]) for o in obs]
        self.actor.fit(obs + [rews, prev_allocs], [allocs],
                       batch_size=self.batch_size, shuffle=True,
                       epochs=self.epochs, verbose=False)
    
    def run(self, episodes, verbose=False, test_run=False):
        episode = 0
        reward_history = []
        end_test=False

        # reset the environment
        observations = self.env.reset()

        # Collect a batch of samples
        while episode < episodes:
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
            
            previous_alloc_vector = self.env.allocation
            tmp_batch['previous_allocation_vector'].append(previous_alloc_vector)

            # While we don't hit the buffer size with our master batch...
            while len(batch['observation']) < self.buffer_size and not end_test:
                # Get the action (scalar), action vector (one-hot vector), 
                # and probability distribution (vector) from the current observation
                alloc_vector = self.get_allocation(observations, previous_alloc_vector)[0]

                # Get the next observation, reward, done, and info for taking an action
                next_observations, rewards, done, info = self.env.step(alloc_vector)

                # Append the data to the mini batch
                tmp_batch['observation'].append(observations)
                tmp_batch['allocation_vector'].append(alloc_vector)
                tmp_batch['previous_allocation_vector'].append(previous_alloc_vector)
                tmp_batch['reward'].append(rewards)

                # The current observation is now the 'next' observation
                observations = next_observations
                previous_alloc_vector = alloc_vector
                
                if test_run:
                    self.env.render()

                # if the episode is at a terminal state...
                if done:
                    # log some reward data (for plotting)
                    reward_data = np.sum(tmp_batch['reward'])
                    reward_history.append(reward_data)

                    # transform rewards based to discounted cumulative rewards
                    # for j in range(len(tmp_batch['reward']) - 2, -1, -1):
                    #     tmp_batch['reward'][j] += tmp_batch['reward'][j + 1] * self.gamma

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

                    # every 10th episode, log some stuff
                    if (episode + 1) % 25 == 0:
                        print('Episode:', episode)
                        print('Reward :', reward_data)
                        print('Average:', np.mean(reward_history[-25:]))
                        print('-'*10)
                        print()
                        self.env.render()

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
                    episode += 1
                    
                    if test_run:
                        end_test = True

            if test_run:
                break
                
            # we've filled up our master batch, so we unpack it into numpy arrays
            _observations = np.array(batch['observation'])
            _allocs = np.array(batch['allocation_vector'])
            _prev_allocs = np.array(batch['previous_allocation_vector'])
            _rewards = np.array(batch['reward'])    
            #rewards = np.reshape(rewards, (len(batch['reward']), 1))

            # train the agent on the batched data
            self.train_batch(_observations, _allocs, _rewards, _prev_allocs)
                    
        self.reward_history = reward_history
        return self.reward_history
```

First, we test the MetaAgent by running an episode on untrained SubAgents and an untrained MetaAgent.


```python
K.clear_session()
sub_envs = []
for i in range(3):
    sub_envs.append(SubEnvironment(noise=0.3))
    
meta_env = MetaEnvironment(sub_envs)
meta_agent = MetaAgent(meta_env)
meta_agent.run(1, test_run=True)
```

    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_1.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_3.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_5.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_7.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_9.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_11.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_13.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_15.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_17.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_19.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_21.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_23.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_25.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_27.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_29.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_31.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_33.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_35.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_37.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_39.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_41.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_43.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_45.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_47.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_49.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_51.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_53.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_55.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_57.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_59.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_61.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_63.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_65.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_67.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_69.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_71.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_73.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_75.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_77.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_79.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_81.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_83.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_85.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_87.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_89.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_91.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_93.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_95.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_97.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_99.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_101.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_103.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_105.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_107.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_109.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_111.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_113.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_115.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_117.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_119.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_121.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_123.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_125.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_127.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_129.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_131.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_133.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_135.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_137.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_139.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_141.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_143.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_145.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_147.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_149.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_151.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_153.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_155.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_157.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_159.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_161.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_163.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_165.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_167.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_169.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_171.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_173.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_175.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_177.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_179.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_181.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_183.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_185.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_187.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_189.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_191.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_193.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_195.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_197.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_199.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_201.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_203.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_205.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_207.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_209.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_211.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_213.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_215.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_217.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_219.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_221.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_223.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_225.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_227.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_229.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_231.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_233.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_235.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_237.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_239.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_241.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_243.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_245.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_247.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_249.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_251.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_253.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_255.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_257.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_259.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_261.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_263.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_265.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_267.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_269.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_271.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_273.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_275.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_277.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_279.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_281.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_283.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_285.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_287.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_289.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_291.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_293.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_295.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_297.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_299.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_301.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_303.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_305.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_307.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_309.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_311.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_313.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_315.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_317.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_319.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_321.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_323.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_325.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_327.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_329.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_331.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_333.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_335.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_337.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_339.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_341.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_343.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_345.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_347.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_349.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_351.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_353.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_355.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_357.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_359.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_361.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_363.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_365.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_367.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_369.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_371.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_373.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_375.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_377.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_379.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_381.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_383.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_385.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_387.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_389.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_391.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_393.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_395.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_397.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_399.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_401.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_403.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_405.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_407.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_409.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_411.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_413.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_415.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_417.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_419.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_421.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_423.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_425.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_427.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_429.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_431.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_433.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_435.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_437.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_439.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_441.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_443.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_445.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_447.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_449.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_451.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_453.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_455.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_457.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_459.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_461.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_463.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_465.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_467.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_469.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_471.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_473.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_475.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_477.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_479.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_481.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_483.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_485.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_487.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_489.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_491.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_493.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_495.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_497.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_499.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_501.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_503.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_505.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_507.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_509.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_511.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_513.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_515.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_517.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_519.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_521.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_523.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_525.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_527.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_529.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_531.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_533.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_535.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_537.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_539.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_541.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_543.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_545.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_547.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_549.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_551.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_553.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_555.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_557.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_559.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_561.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_563.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_565.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_567.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_569.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_571.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_573.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_575.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_577.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_579.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_581.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_583.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_585.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_587.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_589.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_591.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_593.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_595.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_597.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_599.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_601.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_603.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_605.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_607.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_609.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_611.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_613.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_615.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_617.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_619.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_621.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_623.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_625.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_627.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_629.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_631.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_633.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_635.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_637.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_639.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_641.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_643.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_645.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_647.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_649.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_651.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_653.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_655.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_657.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_659.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_661.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_663.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_665.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_667.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_669.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_671.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_673.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_675.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_677.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_679.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_681.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_683.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_685.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_687.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_689.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_691.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_693.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_695.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_697.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_699.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_701.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_703.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_705.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_707.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_709.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_711.png)


    Episode: 0
    Reward : -1.5356403414901816
    Average: -1.5356403414901816
    ----------
    
    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_713.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_715.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_717.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_24_719.png)





    [-1.5356403414901816]



Next, we train the SubAgents and run a test with them.


```python
K.clear_session()
sub_envs = []
for i in range(3):
    sub_envs.append(SubEnvironment(noise=0.3, seed=i))
    
meta_env = MetaEnvironment(sub_envs, seed=1)

for i, agent in enumerate(meta_env.agents):
    print('Training SubAgent {}'.format(i))
    agent.run(100, verbose=False)
    
meta_agent = MetaAgent(meta_env)
meta_agent.run(1, test_run=True)
```

    Training SubAgent 0
    Training SubAgent 1
    Training SubAgent 2
    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_1.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_3.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_5.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_7.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_9.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_11.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_13.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_15.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_17.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_19.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_21.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_23.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_25.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_27.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_29.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_31.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_33.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_35.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_37.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_39.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_41.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_43.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_45.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_47.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_49.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_51.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_53.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_55.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_57.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_59.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_61.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_63.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_65.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_67.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_69.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_71.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_73.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_75.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_77.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_79.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_81.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_83.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_85.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_87.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_89.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_91.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_93.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_95.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_97.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_99.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_101.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_103.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_105.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_107.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_109.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_111.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_113.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_115.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_117.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_119.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_121.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_123.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_125.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_127.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_129.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_131.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_133.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_135.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_137.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_139.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_141.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_143.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_145.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_147.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_149.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_151.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_153.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_155.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_157.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_159.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_161.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_163.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_165.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_167.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_169.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_171.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_173.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_175.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_177.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_179.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_181.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_183.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_185.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_187.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_189.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_191.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_193.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_195.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_197.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_199.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_201.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_203.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_205.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_207.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_209.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_211.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_213.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_215.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_217.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_219.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_221.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_223.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_225.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_227.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_229.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_231.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_233.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_235.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_237.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_239.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_241.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_243.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_245.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_247.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_249.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_251.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_253.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_255.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_257.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_259.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_261.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_263.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_265.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_267.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_269.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_271.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_273.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_275.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_277.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_279.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_281.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_283.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_285.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_287.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_289.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_291.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_293.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_295.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_297.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_299.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_301.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_303.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_305.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_307.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_309.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_311.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_313.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_315.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_317.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_319.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_321.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_323.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_325.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_327.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_329.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_331.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_333.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_335.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_337.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_339.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_341.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_343.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_345.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_347.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_349.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_351.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_353.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_355.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_357.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_359.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_361.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_363.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_365.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_367.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_369.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_371.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_373.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_375.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_377.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_379.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_381.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_383.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_385.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_387.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_389.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_391.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_393.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_395.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_397.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_399.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_401.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_403.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_405.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_407.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_409.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_411.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_413.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_415.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_417.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_419.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_421.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_423.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_425.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_427.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_429.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_431.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_433.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_435.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_437.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_439.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_441.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_443.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_445.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_447.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_449.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_451.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_453.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_455.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_457.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_459.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_461.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_463.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_465.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_467.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_469.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_471.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_473.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_475.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_477.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_479.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_481.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_483.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_485.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_487.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_489.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_491.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_493.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_495.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_497.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_499.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_501.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_503.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_505.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_507.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_509.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_511.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_513.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_515.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_517.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_519.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_521.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_523.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_525.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_527.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_529.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_531.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_533.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_535.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_537.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_539.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_541.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_543.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_545.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_547.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_549.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_551.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_553.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_555.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_557.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_559.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_561.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_563.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_565.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_567.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_569.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_571.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_573.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_575.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_577.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_579.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_581.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_583.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_585.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_587.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_589.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_591.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_593.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_595.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_597.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_599.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_601.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_603.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_605.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_607.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_609.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_611.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_613.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_615.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_617.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_619.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_621.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_623.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_625.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_627.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_629.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_631.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_633.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_635.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_637.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_639.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_641.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_643.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_645.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_647.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_649.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_651.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_653.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_655.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_657.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_659.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_661.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_663.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_665.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_667.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_669.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_671.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_673.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_675.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_677.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_679.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_681.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_683.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_685.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_687.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_689.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_691.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_693.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_695.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_697.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_699.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_701.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_703.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_705.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_707.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_709.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_711.png)


    Episode: 0
    Reward : 36.414370793664446
    Average: 36.414370793664446
    ----------
    
    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_713.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_715.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_717.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_26_719.png)





    [36.414370793664446]



Finally, we first train the SubAgents...


```python
K.clear_session()
sub_envs = []
for i in range(3):
    sub_envs.append(SubEnvironment(noise=0.3, seed=i))
    
meta_env = MetaEnvironment(sub_envs, seed=1)

for i, agent in enumerate(meta_env.agents):
    print('Training SubAgent {}'.format(i))
    agent.run(100, verbose=False)
```

    Training SubAgent 0
    Training SubAgent 1
    Training SubAgent 2


... then we train the MetaAgent on the trained SubAgents.


```python
meta_agent = MetaAgent(meta_env)
rh = meta_agent.run(100)
```

    Episode: 24
    Reward : 42.40901197860731
    Average: 42.026645740677665
    ----------
    
    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_1.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_3.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_5.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_7.png)


    Episode: 49
    Reward : 38.59488688902549
    Average: 41.56749680040769
    ----------
    
    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_9.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_11.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_13.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_15.png)


    Episode: 74
    Reward : 44.07016777251303
    Average: 41.78650803574718
    ----------
    
    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_17.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_19.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_21.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_23.png)


    Episode: 99
    Reward : 42.40901197860731
    Average: 41.72006180399096
    ----------
    
    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_25.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_27.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_29.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_30_31.png)


Then, we test it.


```python
meta_agent.run(1, test_run=True)
```

    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_1.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_3.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_5.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_7.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_9.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_11.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_13.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_15.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_17.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_19.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_21.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_23.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_25.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_27.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_29.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_31.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_33.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_35.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_37.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_39.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_41.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_43.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_45.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_47.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_49.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_51.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_53.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_55.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_57.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_59.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_61.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_63.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_65.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_67.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_69.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_71.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_73.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_75.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_77.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_79.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_81.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_83.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_85.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_87.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_89.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_91.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_93.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_95.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_97.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_99.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_101.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_103.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_105.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_107.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_109.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_111.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_113.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_115.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_117.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_119.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_121.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_123.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_125.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_127.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_129.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_131.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_133.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_135.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_137.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_139.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_141.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_143.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_145.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_147.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_149.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_151.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_153.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_155.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_157.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_159.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_161.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_163.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_165.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_167.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_169.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_171.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_173.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_175.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_177.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_179.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_181.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_183.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_185.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_187.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_189.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_191.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_193.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_195.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_197.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_199.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_201.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_203.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_205.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_207.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_209.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_211.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_213.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_215.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_217.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_219.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_221.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_223.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_225.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_227.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_229.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_231.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_233.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_235.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_237.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_239.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_241.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_243.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_245.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_247.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_249.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_251.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_253.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_255.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_257.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_259.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_261.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_263.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_265.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_267.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_269.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_271.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_273.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_275.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_277.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_279.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_281.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_283.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_285.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_287.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_289.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_291.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_293.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_295.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_297.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_299.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_301.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_303.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_305.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_307.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_309.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_311.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_313.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_315.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_317.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_319.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_321.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_323.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_325.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_327.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_329.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_331.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_333.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_335.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_337.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_339.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_341.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_343.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_345.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_347.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_349.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_351.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_353.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_355.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_357.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_359.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_361.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_363.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_365.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_367.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_369.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_371.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_373.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_375.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_377.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_379.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_381.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_383.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_385.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_387.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_389.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_391.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_393.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_395.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_397.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_399.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_401.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_403.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_405.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_407.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_409.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_411.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_413.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_415.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_417.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_419.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_421.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_423.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_425.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_427.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_429.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_431.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_433.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_435.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_437.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_439.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_441.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_443.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_445.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_447.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_449.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_451.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_453.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_455.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_457.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_459.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_461.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_463.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_465.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_467.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_469.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_471.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_473.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_475.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_477.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_479.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_481.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_483.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_485.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_487.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_489.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_491.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_493.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_495.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_497.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_499.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_501.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_503.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_505.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_507.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_509.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_511.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_513.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_515.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_517.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_519.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_521.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_523.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_525.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_527.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_529.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_531.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_533.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_535.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_537.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_539.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_541.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_543.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_545.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_547.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_549.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_551.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_553.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_555.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_557.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_559.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_561.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_563.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_565.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_567.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_569.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_571.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_573.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_575.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_577.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_579.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_581.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_583.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_585.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_587.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_589.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_591.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_593.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_595.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_597.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_599.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_601.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_603.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_605.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_607.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_609.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_611.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_613.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_615.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_617.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_619.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_621.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_623.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_625.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_627.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_629.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_631.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_633.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_635.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_637.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_639.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_641.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_643.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_645.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_647.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_649.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_651.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_653.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_655.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_657.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_659.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_661.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_663.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_665.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_667.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_669.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_671.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_673.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_675.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_677.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_679.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_681.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_683.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_685.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_687.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_689.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_691.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_693.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_695.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_697.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_699.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_701.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_703.png)


    Sub 0



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_705.png)


    Sub 1



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_707.png)


    Sub 2



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_709.png)


    Meta



![png](Agent-Resource-Allocation_files/Agent-Resource-Allocation_32_711.png)





    [40.57139314014966]




```python

```
