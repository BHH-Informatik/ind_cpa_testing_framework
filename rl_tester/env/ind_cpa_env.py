#! /usr/bin/env python3

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Discrete, MultiDiscrete

class IndCpaEnv(gym.Env):
    '''
    Class containing the custom environment for the IND-CPA testing model \n
    `encryption_algorithm` takes the algorithm which is to be tested \n
    `public_key_length` describes the length of the public keys \n
    `plaintext_length` describes the length of the generated plaintexts \n
    `ciphertext_length` describes the length of the generated ciphertexts \n
    `max_steps` means the number of steps per episode
    '''

    # metadata = {"render_modes": ["human", "ansi", "ansi_list"], "render_fps": 4} # idk what render fps means, do i even need metadata?

    def __init__(self,
                 encryption_algorithm,
                 public_key_length=128,
                 plaintext_length=128,
                 ciphertext_length=128,
                 max_steps=100):
        super().__init__()

        self.public_key_length = public_key_length
        self.plaintext_lenth = plaintext_length
        self.ciphertext_length = ciphertext_length
        self.max_steps = max_steps # max steps per episode
        self.encryption_algorithm = encryption_algorithm

        # observation space
        '''
        contains information the agent can use, in this case key + two pub/priv key pairs
        '''
        self.observation_space = spaces.Dict(
            {"public_key": MultiDiscrete([256]*public_key_length),  # Public key space, e.g., 256-bit key
             "plaintexts": MultiDiscrete([256]*2*plaintext_length), # Two plaintexts: P0 and P1
             "ciphertext": MultiDiscrete([256]*ciphertext_length)}  # Challenge ciphertext
        )

        # action space
        '''
        contains actions the agent can take, in this case choosing two plaintexts
        make guess contains decission, if the chance to guess the right key is above 50%
        '''
        self.action_space = spaces.Dict(
            {"query_plaintexts": spaces.MultiBinary(2 * plaintext_length),
             "choose_plaintexts": MultiDiscrete([256] * 2 * plaintext_length),  # Choose P0 and P1
             "guess": Discrete(2)}  # Guess if ciphertext is P0 or P1
        )

        # placeholders for env state
        self.public_key = None
        self.ciphertext = None
        self.plaintexts = None
        self.challenge_bit = None  # determines if P0 or P1 is encrypted
        self.current_step = 0

    def reset(self):
        '''
        Method for resetting the environment to the default state at initialization of a new episode
        '''
        self.public_key = self.generate_key()
        self.plaintexts = np.zeros((2, self.plaintext_lenth))
        self.ciphertext = np.zeros(self.ciphertext_length)
        self.current_step = 0

        return {"public_key": self.public_key,
                "plaintexts": self.plaintexts,
                "ciphertext": self.ciphertext}
    
    def step(self, action):
        '''
        Method containing main logic of the environment, in this case: \n
        - random encription of plaintext
        - guess of agent, which text is the correct one
        - observations
        - conditions for termination
        '''
        chosen_plaintexts = action["choose_plaintexts"].reshape(2, self.plaintext_lenth)
        self.plaintexts = chosen_plaintexts

        # random encription of plaintext
        self.challenge_bit = np.random.choice([0, 1])
        self.ciphertext = self.encrypt(self.plaintexts[self.challenge_bit], self.generate_key())

        # guess of agent
        guess = action["guess"]
        reward = 1 if guess == self.challenge_bit else 0

        # observations
        observations = {"public_key": self.public_key,
                        "plaintexts":self.plaintexts,
                        "ciphertext": self.ciphertext}

        # termination after allowed number of guesses have been made
        self.current_step += 1

        if self.current_step >= self.max_steps:
            terminated = True
        else:
            terminated = False

        return observations, reward, terminated, {}, {}

    def generate_key(self):
        '''
        Helper method for generating new public keys, calls the suppled encryption algorithm
        '''
        return self.encryption_algorithm.generate_keys()
    
    def encrypt(self, plaintext, public_key):
        '''
        Helper method for encripting a plaintext, calls the supplied encryption algorithm
        '''
        return self.encryption_algorithm.encrypt(plaintext, public_key)
