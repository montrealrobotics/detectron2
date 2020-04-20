import numpy as np
import glob

data_paths = glob.glob('/network/tmp1/bhattdha/detectron2_cityscapes/resnet-101_cityscapes/embeddings_storage/*.npy')
data_paths.sort()

inp = np.load(data_paths[0], allow_pickle=True)[()]

np.random.seed(1000)

indices = np.random.shuffle(np.arange(len(inp['features'])))

features = inp['features'][indices,:]
labels = inp['gt_classes'][indices] 

data_paths.pop(0)

# data_paths = data_paths[:500]

for i, path in enumerate(data_paths):
	print(f"index is {i}")
	inp = np.load(path, allow_pickle=True)[()]
	indices = np.random.shuffle(np.arange(2048))
	fea = inp['features'][indices,:]
	lab = inp['gt_classes'][indices]
	# fea = fea[lab!=3]
	# lab = lab[lab!=3]
	# fea = fea[lab!=4]
	# lab = lab[lab!=4]
	
	features = np.concatenate((features, fea),axis=0)
	labels = np.concatenate((labels, lab),axis=0)

final_dict = {"features":features, "labels": labels}

np.save("/network/tmp1/bhattdha/detectron2_cityscapes/resnet-101_cityscapes/embeddings_storage/final_data.npy", final_dict)

import ipdb; ipdb.set_trace()