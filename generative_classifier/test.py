"""
Not a true test script
"""

import torch.nn.functional as F
from vae import VAE
from train_vae import VAEtrain
import torch
import torchvision
import numpy as np
from torchvision import transforms
# from config import batch_size
from config import *
import random
import glob

input_data = np.load('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data.npy', allow_pickle=True)[()]

## the dataset
X = input_data['features']
y = input_data['labels']

## total reprodicibility
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


models = glob.glob('/network/home/bhattdha/detectron2/generative_classifier/vae_logs_1e-3_32/*.model')
models.sort()
## testing for car for now

indices = np.where(y==1)[0]
data = X[indices,:]

# data = np.random.permutation(data)

train_data_samples = int(len(data)*train_fraction)
val_data_samples = int(len(data) - train_data_samples)

train_data = data[:train_data_samples,:]
val_data = data[train_data_samples:, :]

models_list = []

for model_name in models:
	print(model_name)
	checkpoint = torch.load(model_name)
	model = VAE(latent_space_dim = latent_space_dim).to(device)
	model.load_state_dict(checkpoint['model'])
	model.eval()
	models_list.append(model)



TP = 0
FP = 0
for i in range(0,len(val_data)):

	print(f"At data point {i}/{len(val_data)}")
	## input data
	input_data = torch.from_numpy(val_data[i,:]).float().to(device).view(1, -1)
	
	elbos = []

	for model in models_list:

		with torch.no_grad():
			## the computation happens here
			generated_images, mu, logvar = model(input_data, inference=False)
			reconstruction_error = F.mse_loss(generated_images, input_data, reduction='sum')	
			kldivergence = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  		
			loss =  reconstruction_error + kldivergence
			elbos.append(kldivergence)

	
	## here we check the stuff
	if np.array(elbos).argmin() == 3:
		TP += 1
	else: 
		FP += 1

import ipdb; ipdb.set_trace()