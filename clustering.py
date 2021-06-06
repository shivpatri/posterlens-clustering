import numpy as np
import os
import random
import time
import sklearn.cluster
import sklearn.decomposition
import sklearn.mixture
from scipy.spatial.distance import pdist, cdist
from sklearn.neighbors.nearest_centroid import NearestCentroid

""" Initialization Parameters """
image_path = './embedding/'
data_size = 10000
embedding_size = 512
cluster_count = [3, 5, 7, 10, 15, 20]
random_state = 10
random.seed(random_state)

""" Loading Data """
data = np.zeros((data_size, embedding_size))
files = random.sample(os.listdir(image_path), data_size)
for i, file in enumerate(files):
    data[i] = np.load(image_path + file)

""" Remove Duplicates """
data = np.unique(data, axis=0)

""" Dimensionality reduction using PCA """
data = sklearn.decomposition.PCA(n_components=50).fit_transform(data)

""" Training Data """
kmeans, gm, h = [], [], []
labels = np.zeros((data.shape[0], 18))
for i in range(len(cluster_count)):
    start = time.time()
    kmeans.append(sklearn.cluster.KMeans(n_clusters=cluster_count[i], random_state=random_state, max_iter=100))
    kmeans[-1].fit(data)
    labels[:, i] = kmeans[-1].predict(data)
    gm.append(sklearn.mixture.GaussianMixture(n_components=cluster_count[i], random_state=random_state, max_iter=100))
    gm[-1].fit(data)
    labels[:, i+6] = gm[-1].predict(data)
    h.append(sklearn.cluster.AgglomerativeClustering(n_clusters=cluster_count[i]))
    h[-1].fit(data)
    labels[:, i+12] = h[-1].labels_
    stop = time.time()
    print('Cluster ' + str(cluster_count[i]) + ' took ' + str((stop - start)/60) + ' minutes')

#Agglomerative Clustering doesn't generate cluster centroids, so we generate them manually
h_means = [np.zeros(data.shape)]*len(cluster_count)
for i in range(len(cluster_count)):
        h_means[i] = NearestCentroid().fit(data, h[i].labels_).centroids_

""" Performance metrics """
performance = np.array([0.0]*(3*len(cluster_count)))
for i in range(len(cluster_count)):
    performance[i] = sum(np.min(cdist(data, kmeans[i].cluster_centers_, 'euclidean'), axis=1)) / data.shape[0]
    performance[i+len(cluster_count)] = sum(np.min(cdist(data, gm[i].means_, 'euclidean'), axis=1)) / data.shape[0]
    performance[i+(2*len(cluster_count))] = sum(np.min(cdist(data, h_means[i], 'euclidean'), axis=1)) / data.shape[0]

""" Data for visualization """
np.save('visual_data.npy', np.concatenate((data[:, :3], labels), axis=1))
np.save('performance.npy', performance)