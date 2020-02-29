import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from train_vae import VAEtrain
import torch
import torchvision
import numpy as np
from torchvision import transforms
# from config import batch_size
from config import *
import random
# edit root which is data the first param

input_data = np.load('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data.npy', allow_pickle=True)[()]

## the dataset
X = input_data['features']
y = input_data['labels']

## total reprodicibility
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


for class_label in class_labels:
	print('Training Class# {}'.format(class_label))

	indices = np.where(y==class_label)[0]
	# indices = np.random.permutation(indices)
	data = X[indices,:]
	class_label_fake = (class_label + 5)%len(class_labels)
	indices_fake = np.where(y==class_label_fake)[0]
	val_other_class_data = X[indices_fake,:]
	# data = np.random.permutation(data)

	train_data_samples = int(len(data)*train_fraction)
	val_data_samples = int(len(data) - train_data_samples)

	train_data = data[:train_data_samples,:]
	val_data = data[train_data_samples:, :]

	data = {'train_data': train_data, 'val_data': val_data}

	print(f'Class is: {class_list[class_label]}, number of train data samples are: {len(train_data)}')
	vae_trainer = VAEtrain(class_id = class_label, class_name = class_list[class_label], epochs = epochs[class_label], milestones = milestones[class_label], data = data) # it would create a logs folder and save into it
	# now training this vae
	vae_trainer.train()

