#!/usr/bin/env python

"""
NOTE: this script needs to be executed in the RoboND environment
"""

# import required packages
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# import the policy class
from policy import *

# import the reinforce algorithm
from monitoring import *

def main():
	
	########################################################
	# Required code from main_01.py
	########################################################

	# spin up the environment
	env = gym.make('CartPole-v0')
	#env.seed(0)
	
	# information on the state and action spaces
	print('observation space:', env.observation_space)
	print('action space:', env.action_space)

	# specify the device to sweet sweet gpu action
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# spin up a policy
	policy = Policy().to(device)

	# load trained agent policy weights into model
	policy.load_state_dict(torch.load('checkpoint.pth'))

	# initialise first state
	state = env.reset()

	# set up render mode for the agent
	img = plt.imshow(env.render(mode='rgb_array'))
	
	# run trained agent
	for t in range(2000):
		action, _ = policy.act(state)
		img.set_data(env.render(mode='rgb_array'))
		plt.axis('off')
		state, reward, done, _ = env.step(action)
		#if done:
		#	break

	env.close()


if __name__ == '__main__':
	main()