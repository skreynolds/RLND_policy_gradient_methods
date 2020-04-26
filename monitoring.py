# import required libraries
import numpy as np
from collections import deque

import torch

def reinforce(env, policy, optimizer, n_episodes=1000,
			  max_t=1000, gamma=1.0, print_every=100):
	
	# initialise lists to store episode scores
	scores_deque = deque(maxlen=100)
	scores = []

	for i_episode in range(1, n_episodes+1):
		
		# initialise episode
		saved_log_probs = [] # the log_prob is used in the loss function
		rewards = [] # this list will hold the reward returned at each time step
		state = env.reset()

		# step through episode time steps
		for t in range(max_t):
			action, log_prob = policy.act(state)
			saved_log_probs.append(log_prob) # capture state-action log prob for each time step 
			state, reward, done, _ = env.step(action)
			rewards.append(reward) # add the time step reward to the list
			if done:
				break

		# capture scores
		scores_deque.append(sum(rewards))
		scores.append(sum(rewards))

		# create list of discounts
		discounts = [gamma**i for i in range(len(rewards)+1)]

		# calculate the total discounted reward Gt
		R = sum([d*r for d,r in zip(discounts, rewards)])

		# initialise the list of policy losses
		policy_loss = []
		
		# create a list of tensor object: log probs multiplied by trajectory reward
		for log_prob in saved_log_probs:
			policy_loss.append(-log_prob * R)

		# create the policy loss function
		policy_loss = torch.cat(policy_loss).sum()

		# use the loss function and the optimizer to train the agent
		optimizer.zero_grad()
		policy_loss.backward()
		optimizer.step()

		# print the learning progress of the agent
		if i_episode % print_every == 0:
			print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
		if np.mean(scores_deque)>=195.0:
			print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
			torch.save(policy.state_dict(), 'checkpoint.pth')
			break

	return scores