#! /usr/bin/env python3
from collections import deque  # used for experience replay memory
import numpy as np

import progressbar
from tensorflow.keras.optimizers import Adam

import agent
from env.ind_cpa_env import IndCpaEnv

class Framework:
    '''
    Main class of the framework for testing IND-CPA \n
    `encryption_algorithm` takes the algorithm which is to be tested \n
    `public_key_length` describes the length of the public keys \n
    `plaintext_length` describes the length of the generated plaintexts \n
    `ciphertext_length` describes the length of the generated ciphertexts \n
    `max_steps` number of steps per episode \n
    `num_of_episodes` total number of episodes during training \n
    `batch_size` size of the batches selected for replay memory \n
    `timesteps_per_episode` number of timesteps in one episode \n
    `gamma` discount factor \n
    `epsilon` value for decision between exporation and exploitation \n
    '''
    def __init__(self,
                 encryption_algorithm,
                 public_key_length:int,
                 plaintext_length:int,
                 ciperhtext_lenth:int,
                 max_steps:int,
                 num_of_episodes:int,
                 batch_size:int,
                 timesteps_per_episode:int,
                 gamma:float,
                 epsilon:float):
        self.environment = IndCpaEnv(encryption_algorithm=encryption_algorithm,
                                     public_key_length=public_key_length,
                                     plaintext_length=plaintext_length,
                                     ciphertext_length=ciperhtext_lenth,
                                     max_steps=max_steps)
        # environment.reset()

        self.optimizer = Adam(learning_rate=0.01)
        self.agent = agent.Agent(self.environment,
                                 self.optimizer,
                                 experience_replay=deque(maxlen=2000),
                                 gamma=gamma,
                                 epsilon=epsilon)

        self.batch_size = batch_size
        self.num_of_episodes = num_of_episodes
        self.timesteps_per_episode = timesteps_per_episode
        self.agent.q_network.summary()

    def training(self):
        '''
        Method for training the model
        '''
        for e in range(self.num_of_episodes):
            state = self.environment.reset()
            state = self.flatten_state(state)  # Flatten state into a single vector
            state = np.array([state])  # Ensure state is a 2D array for the neural network

            # Initialize variables
            total_reward = 0
            terminated = False
            
            bar = progressbar.ProgressBar(maxval=self.timesteps_per_episode/10, widgets=[
                progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()
            ])
            bar.start()
            
            for timestep in range(self.timesteps_per_episode):
                guess = self.agent.act(state)
                
                # Environment expects dictionary with plaintexts and guess
                action = {"choose_plaintexts": np.random.randint(0, 256, 2* self.environment.plaintext_lenth),
                          "guess": guess}

                # Take action in the environment
                next_state, reward, terminated, observations, _ = self.environment.step(action)
                next_state = self.flatten_state(next_state)
                next_state = np.array([next_state])

                # Store experience in replay buffer
                self.agent.store(state, guess, reward, next_state, terminated)
                
                state = next_state
                total_reward += reward

                if terminated:
                    self.agent.align_target_model()
                    break

                if len(self.agent.experience_replay) > self.batch_size:
                    self.agent.retrain(self.batch_size)
                
                if timestep % 10 == 0:
                    bar.update(timestep / 10 + 1)
            
            bar.finish()

            if (e + 1) % 10 == 0:
                print("**********************************")
                print(f"Episode: {e + 1}, Total reward: {total_reward}")
                # environment.render()
                print("**********************************")

    def evaluation(self):
        '''
        Method for evaluating the training results
        '''
        total_epochs = 0
        num_of_episodes = 100

        for episode in range(num_of_episodes):
            # Resets the environment
            state = self.environment.reset()
            state = self.flatten_state(state)
            state = np.array([state])

            epochs = 0
            total_reward = 0
            terminated = False
            
            while not terminated:
                guess = self.agent.act(state)
                action = {"choose_plaintexts": np.random.randint(0, 256, 2* self.environment.plaintext_lenth),
                          "guess": guess}

                # Step in the environment
                next_state, reward, terminated, info, _ = self.environment.step(action)
                next_state = self.flatten_state(next_state)
                state = np.array([next_state])

                total_reward += reward
                epochs += 1

            total_epochs += epochs

        print("**********************************")
        print("Results")
        print("**********************************")
        print(f"Average epochs per episode: {total_epochs / num_of_episodes}")
        print(f"Average rewards per episode: {total_reward / num_of_episodes}")

    def flatten_state(self, state):
        """
        Flatten the observation dictionary into a single vector.
        """
        if isinstance(state, tuple):
            state = state[0]

        # Check if state is a dictionary
        if isinstance(state, dict):
            return np.concatenate([np.array(v).flatten().astype(np.float32) for v in state.values()])
        else:
            # If state is not a dictionary
            return np.array(state).flatten().astype(np.float32)
