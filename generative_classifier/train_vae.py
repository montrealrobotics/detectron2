from vae import VAE
import torch
from torch import optim
from tqdm import tqdm
import os
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from config import *
import random
import numpy as np

## total reprodicibility
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class VAEtrain(object):
	"""here, we will create a VAE object and also train the same model, store it etc etc"""
	def __init__(self, class_id, class_name, data= None, epochs = 200, milestones = [40, 100, 150], save_directory_name = 'vae_diff_val_data'):
		super(VAEtrain, self).__init__()
		self.reconstruction_error = []
		self.kl_divergence = []
		self.val_reconstruction_error = []
		self.val_kl_divergence = []

		self.model = VAE(latent_space_dim = latent_space_dim).to(device)

		self.class_id = class_id
		self.class_name = class_name

		self.optimizer = optim.Adam(self.model.parameters(), lr = lr, weight_decay = 0.01)
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = milestones, gamma=0.1, last_epoch=-1)
		
		self.data = data
		assert self.data != None ## there is no training data, bad!
		
		self.train_data = data['train_data']
		self.val_data = data['val_data']

		self.batch_size = batchsize		

		self.validate = True
		self.n_epochs = epochs

		self.iterations = len(self.train_data) * self.n_epochs

		if save_directory_name == None:
			save_directory_name = 'vae_logs'

		self.model_name = 'vae_{}'.format(class_name)

		self.save_directory_name = save_directory_name

		if not(os.path.isdir(self.save_directory_name)):
			print(f"Making dir {self.save_directory_name}")
			os.mkdir(self.save_directory_name)

	def calculate_loss(self, original_image, generated_image, mu, logvar):
		# batch_size = generated_image.shape[0]

		## there are two terms involved in VAE loss
		reconstruction_error = F.mse_loss(generated_image, original_image, reduction='sum')

		## getting kl divergence between predicted distribution and standard normal distribution
		kldivergence = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  

		loss = reconstruction_error + kldivergence

		return loss, reconstruction_error, kldivergence


	def load(self, model_path):
		
		if os.path.isfile(model_path):
			checkpoint = torch.load(model_path)
			self.model.load_state_dict(checkpoint['model'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.reconstruction_error = checkpoint['reconstruction_error']
			self.val_reconstruction_error = checkpoint['val_reconstruction_error']
			self.kl_divergences = checkpoint['kl_divergences']
			self.val_kl_divergences = checkpoint['val_kl_divergences']
			first_epoch = checkpoint['epoch']
			print('model loaded from {}'.format(model_path))
			return first_epoch
		else:
			print('Model {} not found'.format(model_path))
			return None

	def train(self):

		model_name_save = '{}/{}.model'.format(self.save_directory_name, self.model_name)

		if os.path.isfile(model_name_save):
			first_epoch  = self.load(model_name_save)
			if first_epoch is None:
				## don't do shit! 
				first_epoch = 0
			elif first_epoch > (n_epochs - 1):
				print('The model loaded from checkpoint {}'.format(vae_model_name))
				return 
			else:
				print(f"Found pretrained model, starting the training from epoch number {first_epoch}.")

		else:
			print("No checkpoint found, starting from 0.")
			first_epoch = 0

		## let's begin the training buddy
		for epoch in range(first_epoch, self.n_epochs):

			## some statistics to save that you don't know yet

			running_loss = 0

			## training
			for i in range(0,len(self.train_data),self.batch_size):

				## resetting model gradients
				self.model.zero_grad()

				## let's get the input
				input_data = torch.from_numpy(self.train_data[i:(i+batchsize),:]).float().to(device)

				## let's do it
				generated_images, mu, logvar = self.model(input_data,inference=False)

				loss, reconstruction_error, kld = self.calculate_loss(generated_images, input_data, mu, logvar)
				loss.backward()
				self.optimizer.step()
				
				running_loss += loss.item()

				## let's store the statistics
				self.kl_divergence.append(kld.item())
				self.reconstruction_error.append(reconstruction_error.item())


			if self.validate:
				validation_loss = 0

				for i in range(0,len(self.val_data), self.batch_size):

					## let's validate the things
					with torch.no_grad():
						input_data = torch.from_numpy(self.val_data[i:(i+self.batch_size),:]).float().to(device)
						## let's do it
						generated_images, mu, logvar = self.model(input_data,inference=False)
						loss, reconstruction_error, kld = self.calculate_loss(generated_images, input_data, mu, logvar)

						validation_loss += loss.item()
						self.val_kl_divergence.append(kld.item())
						self.val_reconstruction_error.append(reconstruction_error.item())

				print(f"Epoch, training loss and Validation loss are: {epoch + 1, running_loss / len(self.train_data),  validation_loss / len(self.val_data) }")

			else:
				print(f"Epoch and training lossare: {epoch + 1, running_loss / len(self.train_data) }")

			self.lr_scheduler.step()

			"""
			store the state of lr_schedular also, so it knows with 
			what learning rate it should  start training
			"""

			## saving the model and other stuff
			torch.save({
					'model': self.model.state_dict(),
					'epoch': epoch,
					'optimizer': self.optimizer.state_dict(),
					'kl_divergences': self.kl_divergence,
					'reconstruction_error': self.reconstruction_error,
					'val_reconstruction_error': self.val_reconstruction_error,
					'val_kl_divergences': self.val_kl_divergence
				},model_name_save)

