import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

dists = np.load('/home/mila/b/bhattdha/detectron2/unigaussians_loss_att_model_cityscapes.npy')
dists = dists.flatten()

df = 100

dists = np.random.permutation(dists)

samples = []

for i in range(500):
	dists = np.random.permutation(dists)
	for j in np.arange(0, len(dists) - df, df):
		# print('i is: ', i)
		dist_vals = dists[j:(j+df)]
		samples.append(dist_vals.sum())

samples = np.array(samples)
sns.displot(samples, kind="kde")
plt.savefig("unigaussians_loss_att_model_cityscapes.npy") 

mean = samples.mean()
variance = samples.var()

mu1 = df
mu2 = mean
var1 = 2*df
var2 = variance

actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

kldivergence = torch.distributions.kl.kl_divergence(our_dist, actual_dist) 
bhattcharya_distance = 0.25 * np.log(0.25 * (var1/var2 + var2/var1 + 2)) + 0.25 * ((mu1 - mu2)**2/(var1 + var2))
print("Mean and variance are: {} {}".format(mean, variance))
print("kldivergence and Bhattaycharya distances are {} {}".format(kldivergence, bhattcharya_distance))
# import ipdb; ipdb.set_trace()