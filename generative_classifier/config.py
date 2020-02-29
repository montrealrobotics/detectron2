
"""
Okay for now, but make this less shitty in future! 
"""

import torch

class_list = ['Car', 'Van', 'Truck', 'Tram']
epochs = [200, 500, 800, 1000] ## different epochs for different classes
milestones = [[50, 100, 170], [125, 250, 400], [200, 400, 680], [250, 500, 850]]
num_classes = len(class_list) 

class_labels = [i for i in range(num_classes)]
batchsize = 128
train_fraction = 0.8
lr = 1e-3

seed = 100
latent_space_dim = 32	## this is a latent space dimension
n_epochs = 100

if torch.cuda.is_available():
	device = torch.device('cuda:0') # so we can do .to(device)
	print('Using GPU.')
else:
	device = torch.device('cpu')
	print('Using CPU.')