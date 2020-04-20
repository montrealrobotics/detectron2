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
import h5py
# edit root which is data the first param

input_data = h5py.File('/network/tmp1/bhattdha/detectron2_cityscapes/resnet-101_cityscapes/embeddings_storage/final_data.h5', 'r')
# input_data_OOD = np.load('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data_OOD.npy', allow_pickle=True)[()]


## the dataset
X_org = np.array(input_data.get('features'))
y_org = np.array(input_data.get('gt_classes'))
# X_ood = input_data_OOD['features']
# y_ood = input_data_OOD['labels']

# y_ood[y_ood == 6] = 5
# y_ood[y_ood == 7] = 5
# ood_class = [5, 6, 7]
# X =  np.concatenate((X_org, X_ood), axis=0)
# y =  np.concatenate((y_org, y_ood), axis=0)
X = X_org
y = y_org
## total reprodicibility
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

val_data_all_classes = {}
train_data_all_classes = {}

means = {}
covars = {}

train_fraction = 0.8
num_classes = max(y) + 1

class_labels = np.random.permutation(num_classes)
class_labels = np.random.permutation(class_labels)


for class_label in class_labels:
	print('Training Class# {}'.format(class_label))

	indices = np.where(y==class_label)[0]
	# indices = np.random.permutation(indices)
	data = X[indices,:]
	# class_label_fake = (class_label + 5)%len(class_labels)
	# indices_fake = np.where(y==class_label_fake)[0]
	# val_other_class_data = X[indices_fake,:]
	# data = np.random.permutation(data)

	train_data_samples = int(len(data)*train_fraction)
	val_data_samples = int(len(data) - train_data_samples)

	train_data = 1e2*data[:train_data_samples,:]
	val_data = 1e2*data[train_data_samples:, :]

	# data = {'train_data': train_data, 'val_data': val_data}

	val_data_all_classes[str(class_label)] = val_data
	train_data_all_classes[str(class_label)] = train_data
	
	mean = np.mean(train_data, axis = 0)
	cov = np.cov(train_data.T)
	means[str(class_label)] = mean

	covars[str(class_label)] = np.linalg.inv(cov + 1e-10*np.identity(1024))

## let's do the evaluation
stats = {}
maha_dists = {}
for class_label in train_data_all_classes.keys():
	print("Class label: ", class_label)
	data = train_data_all_classes[class_label]
	maha_dists[class_label] = []
	diff = (data - means[class_label])
	maha_dists[class_label] = np.sum(np.multiply(diff, np.dot(covars[class_label], diff.T).T), axis=1)

maha_thresh = {}
gaussian_stats = {'means': means, 'covars': covars}

acc_threshs = [80.0, 85.0, 90.0, 95.0, 98.0, 99.0]
for acc_thresh in acc_threshs:
	acc_classes = {}
	for class_label in train_data_all_classes.keys():
		md_list = maha_dists[class_label]
		md_list.sort()
		index = int(len(md_list)*acc_thresh/100.0)
		acc_classes[class_label] = md_list[index]
		
	maha_thresh[int(acc_thresh)] = acc_classes

final_dict = {'maha_thresh':maha_thresh, 'gaussian_stats':gaussian_stats}

np.save('maha_thresh_cityscapes.npy', final_dict)
import ipdb; ipdb.set_trace()