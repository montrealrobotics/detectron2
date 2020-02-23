
'''
fully connected variational autoencoder
'''

import torch
import torch.nn as nn
import torchvision
import random
import numpy as np
from config import *

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

class VAE(nn.Module):
	def __init__(self, latent_space_dim):
		"""
		n_feature_maps_d to specify model complexity 32 to replicate the one in paper A
		n_channels for MNIST haing 1 channel or cifar10 with 3  channels
		"""

		self.latent_space_dim = latent_space_dim

		super(VAE, self).__init__()
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

			nn.Linear(64, 2*latent_space_dim),
			nn.BatchNorm1d(2*latent_space_dim),
			nn.ELU(inplace=True)

		)

		"""
		decoder
		"""

		self.decoder = nn.Sequential(
			nn.Linear(latent_space_dim, 64),
			nn.BatchNorm1d(64),
			nn.ELU(inplace=True),
			
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



		# #TODO we need to use upsampling
		# self.noise_latent_dim = noise_latent_dim
		# self.decoder = nn.Sequential(
		#     nn.ConvTranspose2d(noise_latent_dim,32, kernel_size = 4 , stride = 1),
		#     nn.BatchNorm2d(32),
		#     nn.ELU(alpha=1.0,inplace=True),

		#     nn.ConvTranspose2d(32, 16, kernel_size = 5, stride=2),
		#     nn.BatchNorm2d(16),
		#     nn.ELU(alpha=1.0,inplace=True),

		#     nn.ConvTranspose2d(16,16,kernel_size = 5, stride=2),
		#     nn.BatchNorm2d(16),
		#     nn.ELU(alpha=1.0,inplace=True),

		#     nn.ConvTranspose2d(16,1,kernel_size=4,stride=1),

		# )
		
	def get_noise(self):	

		## noise 
		noise = torch.FloatTensor(1,latent_space_dim).to(device)

		return noise.data.normal_()

	def reparameterize(self, mu, logvar):
		## need this because sampling is not differentiable opertaion
		## can have different noise for entire batch too
		std = logvar.mul(0.5).exp_()
		return mu + std * self.get_noise()
	
	def bottleneck(self, h):
		
		## first half is mean and second half is logvariance
		mu = h[:,:latent_space_dim]
		logvar = h[:,latent_space_dim:]

		## sampling from the distribution predicted
		z = self.reparameterize(mu, logvar)

		return z, mu, logvar

	def encode(self, x):
		h = self.encoder(x)
		z, mu, logvar = self.bottleneck(h) 
		return z, mu, logvar
	  
	def decode(self,z):
		return self.decoder(z)

	def forward(self, x, inference=True):

		## now let's encode and sample
		z, mu, logvar = self.encode(x)

		## deconding with sampled variable
		images = self.decode(z)
		
		if inference:
		  return images
		return images, mu, logvar


# x = torch.randn((64, 1024))

# vae = VAE(16)
# vae = vae.cuda()
# x = x.cuda()
# recons, mu, logvar = vae(x, inference=False)
# import ipdb; ipdb.set_trace()