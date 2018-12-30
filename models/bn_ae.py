from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

BN = None
__all__ = ['bn_AE']

class bn_AE(nn.Module):
	def __init__(self, option, evaluation=False, normalize=True):

		global BN

		def BNFunc(*args, **kwargs):
			return SyncBN(*args, **kwargs, group_size=1, group=None, sync_stats=True)

		BN = BNFunc

		super(bn_AE, self).__init__()

		# Encoder
		self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.fc1 = nn.Linear(16 * 16 * 32, option['intermediate_size'])
		self.bn1 = BN(3)
		self.bn2 = BN(32)
		self.bn3 = BN(32)
		self.bn4 = BN(32)

		# Latent space
		self.fc21 = nn.Linear(option['intermediate_size'], option['hidden_size'])

		# Decoder
		self.fc3 = nn.Linear(option['hidden_size'], option['intermediate_size'])
		self.fc4 = nn.Linear(option['intermediate_size'], 8192)
		self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
		self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

		self.evaluation = evaluation
		self.normalize = normalize
	def encode(self, x):
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.relu(self.bn2(self.conv2(out)))
		out = self.relu(self.bn3(self.conv3(out)))
		out = self.relu(self.bn4(self.conv4(out)))
		out = out.view(out.size(0), -1)
		h1 = self.relu(self.fc1(out))
		if self.evaluation:
			return h1, self.fc21(h1)
		return self.fc21(h1)

#	def reparameterize(self, mu, logvar):
#		if self.training:
#			std = logvar.mul(0.5).exp_()
#			eps = Variable(std.data.new(std.size()).normal_())
#			return eps.mul(std).add_(mu)
#		else:
#			return mu

	def decode(self, z):
		h3 = self.relu(self.fc3(z))
		out = self.relu(self.fc4(h3))
		# import pdb; pdb.set_trace()
		out = out.view(out.size(0), 32, 16, 16)
		out = self.relu(self.deconv1(out))
		out = self.relu(self.deconv2(out))
		out = self.relu(self.deconv3(out))
		out = self.sigmoid(self.conv5(out))
		return out

	def forward(self, x):
		if self.evaluation:
			m, mu = self.encode(x)
			if self.normalize:
				norm = torch.norm(m,dim=1).view(m.size(0),1)
				norm = torch.cat([norm]*m.size(1),dim=1)
				m = m/norm
			return m
		mu = self.encode(x)
		return self.decode(mu), mu, mu
