import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import numpy as np
from torchvision import transforms
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from config import *
import random

maha_dists = np.load('maha_dists.npy',allow_pickle=True)
input_data = np.load('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data.npy', allow_pickle=True)[()]
input_data_OOD = np.load('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data_OOD.npy', allow_pickle=True)[()]


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

X_ood = input_data_OOD['features']
y_ood = input_data_OOD['labels']
val_data_all_classes = {}

means = {}
covars = {}

# train_fraction = 0.7
num_classes = max(y)+1

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

maha_class = maha_dists[:,5].astype(int)
maha_true_dist = []
maha_false_dist = []

# for ind, m in enumerate(maha_dists):
# 	maha_true_dist.append(m[maha_class[ind]])
# 	m[maha_class[ind]] = np.inf
# 	m[5] = np.inf
# 	maha_false_dist.append(m.min())


## loading the results
# maha_true_dist = np.array(maha_true_dist)
# maha_false_dist = np.array(maha_false_dist)
# input_data_OOD = np.load('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data_OOD.npy', allow_pickle=True)[()]
# X_ood = input_data_OOD['features']
# y_ood = input_data_OOD['labels']

acc_threshs = [60.0, 70.0, 80.0, 85.0, 90.0, 95.0]
ood_stats = {}
for acc_thresh in acc_threshs:
	print("For accuracy: ", acc_thresh)
	mahathresh = {}
	class_dist = {}
	for i in range(num_classes):
		class_dist[i] = maha_dists[maha_dists[:,5]==i][:,i]
		class_dist[i].sort()
		class_dist[i] = class_dist[i][::-1]
		index = int(len(class_dist[i]) - len(class_dist[i])*acc_thresh/100.0)
		mahathresh[i] = class_dist[i][index]

	# mahathresh = {0: 3093.944707607109, 1: 5710.413855647991, 2: 28235.425795092746, 3: 79163.39452332728, 4: 2313.9860080440644}
	tp = 0
	fp = 0

	for x in X_ood:
		data_point = 1e2*x
		flag = True
		mds = [] ## has mahalanobis distances
		for mean_label in means.keys():
			diff = (data_point - means[mean_label]).reshape(len(data_point), 1)
			
			mahalanobis_distance = np.dot(diff.T, np.dot(covars[mean_label], diff))[0][0]
			
			# maha_all.append(mahalanobis_distance)
			mds.append(mahalanobis_distance)

		for i in mahathresh.keys():
			
			if mahathresh[i] > mds[i]:
				fp += 1
				flag = False
				break
			else:
				continue
				
		if flag:
			tp += 1

	ood_stats[acc_thresh] = {'tp':tp, 'fp':fp, 'accuracy': tp/(tp+fp)}

import ipdb; ipdb.set_trace()

colors = ['C'+str(i+1) for i in range(5)]

for i in range(4):
	plt.plot(class_dist[i], '-o', alpha=0.7, color=colors[i], label="class maha dists"+str(i).zfill(5))
	# plt.plot(maha_false_dist, '-o', alpha=0.7, color=colors[1], label="maha_false_dist")

# [1e3, 1e4,  1e3]
plt.legend()
plt.legend(loc='upper right')
plt.xlabel('datapoint ->')
plt.ylabel('mahalanobis distance -> ')
plt.title('Mahalanobis distance plot')
plt.savefig('maha_dists.png')




import ipdb; ipdb.set_trace()