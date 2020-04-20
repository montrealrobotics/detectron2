import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torchvision
import numpy as np
from torchvision import transforms
# from config import batch_size
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from config import *
import random
# edit root which is data the first param

input_data = np.load('/network/tmp1/bhattdha/detectron2_cityscapes/resnet-101_cityscapes/embeddings_storage/final_data.npy', allow_pickle=True)[()]
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

tp = 0
fp = 0 

all_scores = {}
all_labels = {}


for class_label in val_data_all_classes.keys():
	all_scores[class_label] = []
	all_labels[class_label] = []
	data = val_data_all_classes[class_label]
	for data_point in data:
		# print(data_point)
		mds = [] ## has mahalanobis distances
		for mean_label in means.keys():
			
			
			diff = (data_point - means[mean_label]).reshape(len(data_point), 1)
			
			mahalanobis_distance = np.dot(diff.T, np.dot(covars[mean_label], diff))[0][0]
			
			# maha_all.append(mahalanobis_distance)
			mds.append(mahalanobis_distance)
		
		all_scores[class_label].append(np.array(mds).min())
		all_labels[class_label].append(0)

mahathresh = {0: 3093.944707607109, 1: 5710.413855647991, 2: 28235.425795092746, 3: 79163.39452332728, 4: 2313.9860080440644}
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

import ipdb; ipdb.set_trace()		


	# for class_label in val_data_all_classes.keys():

	# 	all_scores[class_label].append(np.array(mds).min())
	# 	all_labels[class_label].append(1)
	

# def log_pr_curve(num_classes, log_dir, scores, y, epoch = 0):
#     average_precision = average_precision_score(y, scores)
#     print('Average precision-recall score: {0:0.2f}'.format(average_precision))
#     precision, recall, thresholds = precision_recall_curve(y, scores)
#     plt.step(recall, precision, color='b', alpha=0.2,
#              where='post')
#     plt.fill_between(recall, precision, alpha=0.2, color='b')#, **step_kwargs)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.title('2-class Precision-Recall curve AP={0:0.2f}'.format(average_precision))
#     plt.savefig(os.path.join(log_dir, 'pr_' + str(epoch) + '.png'))
#     plt.clf()


for class_label in val_data_all_classes.keys():
	all_scores_class = np.clip(np.array(all_scores[class_label]), 0, 1e5)/1e5
	all_labels_class = np.array(all_labels[class_label])
	log_pr_curve('2', './', all_scores_class, all_labels_class, class_label)











# for class_label in val_data_all_classes.keys():
# 	data = val_data_all_classes[class_label]
# 	tp = 0 
# 	fp = 0
# 	for data_point in data:
# 		# print(data_point)
# 		mds = [] ## has mahalanobis distances
# 		for mean_label in means.keys():
			
			
# 			diff = (data_point - means[mean_label]).reshape(len(data_point), 1)
			
# 			mahalanobis_distance = np.dot(diff.T, np.dot(covars[mean_label], diff))[0][0]
			
# 			# maha_all.append(mahalanobis_distance)
# 			mds.append(mahalanobis_distance)

# 		maha_all.append(mds+[int(class_label)])
		
# 		if str(class_labels[np.argmin(np.array(mds))]) == class_label:
# 			tp += 1
# 		else: 
# 			fp += 1
# 	stats[class_label] = {'tp':tp, 'fp':fp, 'accuracy': 100.0*tp/(tp+fp)}

# np.save('maha_dists.npy', maha_all)

		






