import numpy as np
import glob

data_paths = glob.glob('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/*.npy')
data_paths.sort()

inp = np.load(data_paths[0], allow_pickle=True)[()]

features = inp['features']
labels = inp['gt_classes']

data_paths.pop(0)
# data_paths = data_paths[:500]

for i, path in enumerate(data_paths):
	print(f"index is {i}")
	inp = np.load(path, allow_pickle=True)[()]
	features = np.concatenate((features, inp['features']),axis=0)
	labels = np.concatenate((labels, inp['gt_classes']),axis=0)

final_dict = {"features":features, "labels": labels}

np.save("/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data.npy", final_dict)

import ipdb; ipdb.set_trace()