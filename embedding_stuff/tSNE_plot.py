from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import torch	
# mnist = fetch_mldata("MNIST original")
# X = mnist.data / 255.0
# y = mnist.target

from autoencoder import autoencoder
np.random.seed(42)
autoencoder_obj = autoencoder()

input_data = np.load('/network/tmp1/bhattdha/detectron2_kitti/embeddings_storage/final_data.npy', allow_pickle=True)[()]

X = input_data['features']
y = input_data['labels']

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

X = X_new
y = y_new
# import ipdb; ipdb.set_trace()

print(X.shape, y.shape)

## instead of PCA, we use autoencoders for dimensionality reduction
autoencoder_obj.load_state_dict(torch.load('autoencoder_02.model'))
autoencoder_obj =autoencoder_obj.cuda()
autoencoder_obj.eval()

X = torch.from_numpy(X).cuda()

## forward pass 
X = autoencoder_obj.encode(X)
X = X.detach().cpu().numpy()


feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
# X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))


rndperm = np.random.permutation(df.shape[0])

# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(df[feat_cols].values)
# df['pca-one'] = pca_result[:,0]
# dy = f['pca-two'] = pca_result[:,1] 
# df['pca-three'] = pca_result[:,2]
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# N = 10000
N = len(X)
X = X[rndperm[:N],:]
# import ipdb; ipdb.set_trace()
df_subset = df.loc[rndperm[:N],:].copy()
# data_subset = df_subset[feat_cols].values
# # pca = PCA(n_components=3)
# # pca_result = pca.fit_transform(data_subset)
# # df_subset['pca-one'] = pca_result[:,0]
# # df_subset['pca-two'] = pca_result[:,1] 
# # df_subset['pca-three'] = pca_result[:,2]
# # print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=5, perplexity=5000, n_iter=500)
# tsne_pca_results = tsne.fit_transform(X)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
# # import ipdb;ipdb.set_trace()

# df_subset['tsne-ae-one'] = tsne_pca_results[:,0]
# df_subset['tsne-ae-two'] = tsne_pca_results[:,1]
# plt.figure(figsize=(16,4))
# ax = plt.subplot(1, 1, 1)

# # import ipdb;ipdb.set_trace()

# sns.scatterplot(x="tsne-ae-one", y="tsne-ae-two",hue="y", palette=sns.color_palette("hls", 5),data=df_subset,legend="full", alpha=1.0, ax = ax)

# # fig = sns_plot.get_figure()
# plt.savefig('fig.png')

## let's try umap
import umap
reducer = umap.UMAP(verbose=True, n_neighbors=1000)
print("starting umap")
time_start = time.time()
umap_results = reducer.fit_transform(X)
print('UMAP done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['umap-ae-one'] = umap_results[:,0]
df_subset['umap-ae-two'] = umap_results[:,1]

# import ipdb;ipdb.set_trace()

plt.figure(figsize=(16,4))
ax = plt.subplot(1, 1, 1)


sns.scatterplot(x="umap-ae-one", y="umap-ae-two",hue="y", palette=sns.color_palette("hls", 5),data=df_subset,legend="full", alpha=1.0, ax = ax)

# fig = sns_plot.get_figure()
plt.savefig('umap_fig.png')