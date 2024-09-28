#! /usr/bin/env python3

import random
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class Agent:
    '''
    Class defining the agent interacting with the environment. \n
    `environment` environment for the agent to interact with \n
    `optimizer` optimizer to be used \n
    `experience_replay` replay memory \n
    `gamma` discount factor \n
    `epsilon` value for decision between exporation and exploitation \n
    '''
    def __init__(self,
                 environment,
                 optimizer,
                 experience_replay,
                 gamma,
                 epsilon):
        self.environment = environment
        
        # Initialize atributes
        self._state_size = environment.observation_space
        self._action_size = environment.action_space
        self._optimizer = optimizer

        self.experience_replay = experience_replay

        # Initialize discount and exploration rate
        self.gamma = gamma
        self.epsilon = epsilon

        self._state_size = self.get_state_size()
        self._action_size = 2

        # Build networks
        self.q_network = self.build_and_compile_model()
        self.target_network = self.build_and_compile_model()
        self.align_target_model()

    def get_state_size(self):
        """
        Flatten observation space to use it as input size
        """
        observation_sample = self.environment.reset()  # initial observation

        # handling for unpacking tuple
        if isinstance(observation_sample, tuple):
            observation_sample = observation_sample[0]
    
        # Ensure observation is a dictionary
        if isinstance(observation_sample, dict):
            return sum(np.prod(np.array(v).shape) for v in observation_sample.values())
        else:
            # handling for if observation is array (just in case)
            return np.prod(np.array(observation_sample).shape)

    def flatten_state(self, state):
        """
        Flatten observation dictionary into a single vector
        """
        if isinstance(state, tuple):
            state = state[0]

        # Check if state is a dictionary
        if isinstance(state, dict):
            return np.concatenate([np.array(v).flatten() for v in state.values()])
        else:
            # If state is not a dictionary
            return np.array(state).flatten()

    def store(self, state, action, reward, next_state, terminated):
        """
        Method for storing data in experience replay \n
        Stores data as tupel of '<s,s',a,r>'
        """
        self.experience_replay.append((state, action, reward, next_state, terminated))
    
    def build_and_compile_model(self):
        """
        Main method for building the model itself \n
        Changes to the layout go here
        """
        model = Sequential()
        model.add(Dense(128, input_dim=self._state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def align_target_model(self):
        '''
        Copy weights from the Q-network to the target network
        '''
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        """
        Method for implementing exploration/ exploitation decision (epsilon)
        """
        print(type(state))
        print(state)
        if np.random.rand() <= self.epsilon:
            return random.choice([0,1])
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        """
        Method for trainig the Q- Model \n
        Picks random samples from experience replay memory as training data
        """
        minibatch = random.sample(self.experience_replay, batch_size)
        for state, action, reward, next_state, terminated in minibatch:
            target = self.q_network.predict(state)
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            self.q_network.fit(state, target, epochs=1)
