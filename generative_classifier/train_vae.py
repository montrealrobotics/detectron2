from vae import VAE
import torch
from torch import optim
from tqdm import tqdm
import os
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


import random

## total reprodicibility
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
np.random.seed(100)
random.seed(100)

if torch.cuda.is_available():
	device = torch.device('cuda:0') # so we can do .to(device)
	print('Using GPU.')
else:
	device = torch.device('cpu')
	print('Using CPU.')

class VAEtrain(object):
	"""here, we will create a VAE object and also train the same model, store it etc etc"""
	def __init__(self, class_id, class_name, data= None, save_directory_name = 'vae_logs'):
		super(VAEtrain, self).__init__()
		self.reconstruction_error = []
		self.kl_divergence = []
		self.val_reconstruction_error = []
		self.val_kl_divergence = []

		self.model = VAE().to(device)

		self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-5)
		self.data = data
		self.train_data = None
		self.val_data = None
		self.train_fraction = 0.8

		self.validate = True
		self.n_epochs = 100

		assert self.train_loader != None
		self.iterations = len(train_loader) * self.n_epochs

		if save_directory_name == None:
			save_directory_name = 'logs'

		self.model_name = 'vae_{}'.format(class_name)

		self.save_directory_name = save_directory_name

		if not(os.path.isdir(self.save_directory_name)):
			print(f"Making dir {self.save_directory_name}")
			os.mkdir(self.save_directory_name)

		def calculate_loss(original_image, generated_image, mu, logvar):
			batch_size = generated_image.shape[0]

			## there are two terms involved in VAE loss
			reconstruction_error = F.mse_loss(generated_image, original_image)

			## getting kl divergence between predicted distribution and standard normal distribution
			kldivergence = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  

			loss = reconstruction_error + kldivergence

			return loss, reconstruction_error, kldivergence

		def load(self):
			vae_model_name = '{}/{}.tar'.format(self.saving_folder_name,self.model_name)
			if os.path.isfile(vae_model_name):
				checkpoint = torch.load(vae_model_name)
				self.model.load_state_dict(checkpoint['model'])
				self.optimizer.load_state_dict(checkpoint['optimizer'])
				self.reconstruction_error = checkpoint['reconstruction_error']
				self.val_reconstruction_error = checkpoint['val_reconstruction_error']
				self.kl_divergences = checkpoint['kl_divergences']
				self.val_kl_divergences = checkpoint['val_kl_divergences']
				first_epoch = checkpoint['epoch']
				print('model loaded from {}'.format(vae_model_name))
				return first_epoch
			else:
				print('Model {} not found'.format(self.model_name))

		def train(self):

			for i, data in enumerate(self.train_data)