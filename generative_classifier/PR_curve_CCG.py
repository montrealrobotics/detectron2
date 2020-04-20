import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torchvision
import numpy as np
from torchvision import transforms
# from config import batch_size
from config import *
import random
# edit root which is data the first param

input_data = np.load('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data.npy', allow_pickle=True)[()]
# input_data_OOD = np.load('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data_OOD.npy', allow_pickle=True)[()]


## the dataset
X_org = input_data['features']
y_org = input_data['labels']
# X_ood = input_data_OOD['features']
# y_ood = input_data_OOD['labels']

# y_ood[y_ood == 6] = 5
# y_ood[y_ood == 7] = 5
# ood_class = [5, 6, 7]
X =  X_org
y =  y_org
## total reprodicibility
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

val_data_all_classes = {}

means = {}
covars = {}

# train_fraction = 0.7
num_classes = max(y) + 1

class_labels = np.arange(num_classes)
# class_labels = np.random.permutation(num_classes)
# class_labels = np.random.permutation(class_labels)


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
	
	mean = np.mean(train_data, axis = 0)
	cov = np.cov(train_data.T)
	means[str(class_label)] = mean
	## may be wrong! 
	covars[str(class_label)] = np.linalg.inv(cov + 1e-10*np.identity(1024))


## let's do the evaluation
stats = {}
maha_all = []
for class_label in val_data_all_classes.keys():
	data = val_data_all_classes[class_label]
	tp = 0 
	fp = 0
	for data_point in data:
		# print(data_point)
		mds = [] ## has mahalanobis distances
		for mean_label in means.keys():
			
			
			diff = (data_point - means[mean_label]).reshape(len(data_point), 1)
			
			mahalanobis_distance = np.dot(diff.T, np.dot(covars[mean_label], diff))[0][0]
			
			# maha_all.append(mahalanobis_distance)
			mds.append(mahalanobis_distance)

		maha_all.append(mds+[int(class_label)])
		
		if str(class_labels[np.argmin(np.array(mds))]) == class_label:
			tp += 1
		else: 
			fp += 1
	stats[class_label] = {'tp':tp, 'fp':fp, 'accuracy': 100.0*tp/(tp+fp)}

np.save('maha_dists.npy', maha_all)

		






