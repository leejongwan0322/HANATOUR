import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X= -2*np.random.rand(100,2)
X1 = 1+2*np.random.rand(50,2)
X[50:100, :] = X1
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
# plt.show()

from sklearn.cluster import KMeans
# Kmean = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',random_state=None, tol=0.0001, verbose=0)
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)

print(Kmean.cluster_centers_)
print(Kmean.cluster_centers_[0])
plt.scatter(X[ : , 0], X[ : , 1], s=100, c='b')
plt.scatter(Kmean.cluster_centers_[0][0], Kmean.cluster_centers_[0][1], s=400, c='g', marker='s')
plt.scatter(Kmean.cluster_centers_[1][0], Kmean.cluster_centers_[1][1], s=400, c='r', marker='s')
plt.show()

print(Kmean.labels_)
sample_test=np.array([-3.0,-3.0])
second_test=sample_test.reshape(1, -1)
Kmean.predict(second_test)
print(second_test)