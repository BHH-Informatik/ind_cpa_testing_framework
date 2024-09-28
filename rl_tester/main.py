#! /usr/bin/env python3

import argparse

from algorithms import rsa_vulnerable
import ind_cpa_framework

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm',
                    default='rsa_vulnerable',
                    choices=['rsa_vulnerable'],
                    help='encryption algorithm to be used, default: %(default)s')
parser.add_argument('--public_key_length',
                    type=int,
                    help='length of the public key to be generated')
parser.add_argument('--plaintext_length',
                    type=int,
                    help='length of the plaintexts generated in the testing process')
parser.add_argument('--ciphertext_length',
                    type=int,
                    help='length of the ciphertexts to be generated in the process')
parser.add_argument('--max_steps',
                    type=int,
                    help='number of steps per episode')
parser.add_argument('--num_of_episodes',
                    type=int,
                    help='number of episodes in the training process')
parser.add_argument('--gamma',
                    type=float,
                    help='discount factor to be applied, controls if agent prioritizes reward or value')
parser.add_argument('--epsilon',
                    type=float,
                    help='value influencing the expolration/ exploitation decision')
parser.add_argument('--batch_size',
                    type=int,
                    help='size of the data selected from experience replay')
parser.add_argument('--timesteps_per_episode',
                    type=int,
                    help='number of timesteps in one episode')
args = parser.parse_args()

if __name__ == '__main__':
    match args.algorithm:
        case 'rsa_vulnerable':
            encryption_algorithm = rsa_vulnerable.Algorithm(args.public_key_length)
        case _:
            raise NameError("unknown encryption algorithm")
    
    framework = ind_cpa_framework.Framework(encryption_algorithm=encryption_algorithm,
                                            public_key_length=args.public_key_length,
                                            plaintext_length=args.plaintext_length,
                                            ciperhtext_lenth=args.ciphertext_length,
                                            max_steps=args.max_steps,
                                            num_of_episodes=args.num_of_episodes,
                                            gamma=args.gamma,
                                            epsilon=args.epsilon,
                                            batch_size=args.batch_size,
                                            timesteps_per_episode=args.timesteps_per_episode)
    
    framework.training()
    framework.evaluation()
