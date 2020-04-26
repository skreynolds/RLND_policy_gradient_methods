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
	# Initialise the environment and train the agent
	########################################################

	# spin up the environment
	env = gym.make('CartPole-v0')
	env.seed(0)
	
	# information on the state and action spaces
	print('observation space:', env.observation_space)
	print('action space:', env.action_space)

	# specify the device to sweet sweet gpu action
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# spin up a policy
	policy = Policy().to(device)

	# initialise the torch optimiser
	optimizer = optim.Adam(policy.parameters(), lr=1e-2)

	# train the agent
	scores = reinforce(env, policy, optimizer)

	# plot the learning graphs
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(np.arange(1, len(scores)+1), scores)
	plt.xlabel('Episode #')
	plt.ylabel('Score')
	plt.show()


if __name__ == '__main__':
	main()