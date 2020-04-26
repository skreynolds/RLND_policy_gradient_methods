# import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ensure that the gpu is used where possible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Policy(nn.Module):
	def __init__(self, s_size=4, h_size=16, a_size=2):
		
		# inherit properties from parent class
		super(Policy, self).__init__()

		# specifiy layers that will be used
		self.fc1 = nn.Linear(s_size, h_size)
		self.fc2 = nn.Linear(h_size, a_size)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x, dim=1)

	def act(self, state):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		probs = self.forward(state).cpu()
		m = Categorical(probs)
		action = m.sample()
		return action.item(), m.log_prob(action)