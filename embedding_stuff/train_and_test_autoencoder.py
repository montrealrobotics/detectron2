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
from autoencoder import autoencoder
np.random.seed(42)

## let's define the network
autoencoder_obj = autoencoder()



## counting number of parameters in the model
params_count = sum(p.numel() for p in autoencoder_obj.parameters()) 
print(f"Number of parameters in the model are:{params_count}")	


input_data = np.load('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data.npy', allow_pickle=True)[()]

X = input_data['features']
y = input_data['labels']

# import ipdb; ipdb.set_trace()
indices = np.where(y==3)[0]
X_new = X[indices,:]
y_new = y[indices]

num_samples_per_class = len(y_new)
class_labels = [0,1,2,4]

## get samples for 
for label in class_labels:
	indices = np.where(y==label)[0]
	indices_ind = np.random.permutation(len(indices))[:num_samples_per_class]
	X_new = np.concatenate((X_new, X[indices[indices_ind],:]), axis=0)
	y_new = np.concatenate((y_new, y[indices[indices_ind]]), axis=0)

features = X_new
labels = y_new

# features = X
# labels = y

# features = input_data['features']
# labels = input_data['labels']
# indices = np.where(labels!=5)
# features = features[indices[0],:]

# import ipdb; ipdb.set_trace()

epochs = 1000
batchsize = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

autoencoder_obj = autoencoder_obj.to(device)
autoencoder_obj.train()
criterion = nn.MSELoss()
optimizer = optim.SGD(autoencoder_obj.parameters(), lr=1e-8, momentum=0.9, nesterov=True, weight_decay = 0.01)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10000, 20000, 40000], gamma=0.1, last_epoch=-1)

for epoch in range(epochs):

	running_loss = 0
	## let's train
	for i in range(0,len(features),batchsize):
		
		## let's get the input
		input_data = torch.from_numpy(features[i:(i+batchsize),:]).float().to(device)

		## same as input data
		output = input_data

		## reconstruction
		reconstruction = autoencoder_obj(input_data)

		loss = criterion(output, reconstruction)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
	
	print('[%d, %5d] Train loss: %.10f' %(epoch + 1, i, running_loss * batchsize / len(features) ))
	lr_scheduler.step()
	# print(len(features))


torch.save(autoencoder_obj.state_dict(), "autoencoder_batchnorm.model")

import ipdb; ipdb.set_trace()
