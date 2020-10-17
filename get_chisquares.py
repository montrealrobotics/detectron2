import numpy as np

dists = np.load('/home/mila/b/bhattdha/detectron2/unigaussians.npy')
dists = dists.flatten()

df = 500

dists = np.random.permutation(dists)

samples = []

for i in np.arange(0, len(dists) - df, df):
	print('i is: ', i)
	dist_vals = dists[i:(i+df)]
	samples.append(dist_vals.sum())

samples = np.array(samples)

mean = samples.mean()
variance = samples.var()

print("Mean and variance are: {}".format(mean, variance))
import ipdb; ipdb.set_trace()