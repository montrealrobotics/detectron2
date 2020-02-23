
## let's make a quick autoencoder
import numpy as np
import torch
from torch import nn
import torchvision 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random


class autoencoder(nn.Module):
	def __init__(self):
		super(autoencoder, self).__init__()

		"""
		encoder
		"""

		self.encoder = nn.Sequential(
			nn.Linear(1024, 768),
			nn.BatchNorm1d(768),
			nn.ELU(inplace=True),

			nn.Linear(768, 512),
			nn.BatchNorm1d(512),
			nn.ELU(inplace=True),

			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ELU(inplace=True),

			nn.Linear(256, 128),
			nn.BatchNorm1d(128),
			nn.ELU(inplace=True),

			nn.Linear(128, 64),
			nn.BatchNorm1d(64),
 			nn.ELU(inplace=True),

			# nn.Linear(64, 32),
			# nn.BatchNorm1d(32),
			# nn.ELU(inplace=True)

		)

		"""
		decoder
		"""

		self.decoder = nn.Sequential(
			# nn.Linear(32, 64),
			# nn.BatchNorm1d(64),
			# nn.ELU(inplace=True),
			
			nn.Linear(64, 128),
			nn.BatchNorm1d(128),
			nn.ELU(inplace=True),

			nn.Linear(128, 256),
			nn.BatchNorm1d(256),
			nn.ELU(inplace=True),

			nn.Linear(256, 512),
			nn.BatchNorm1d(512),
			nn.ELU(inplace=True),

			nn.Linear(512, 768),
			nn.BatchNorm1d(768),
			nn.ELU(inplace=True),

			nn.Linear(768, 1024),
			
		)


	def encode(self,x):
		return self.encoder(x)

	def decode(self, x):
		return self.decoder(x)

	def forward(self, x):
		## encoder + decoder
		return self.decode(self.encode(x))


