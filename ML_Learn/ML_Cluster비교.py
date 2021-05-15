from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_circles
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X, y = make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=0, factor=0.5)
clusterDF = pd.DataFrame(data=X, columns=['ftr1', 'ftr2'])
clusterDF['target'] = y


plt.scatter(X[:,0],X[:,1],c=y)
plt.colorbar()
plt.show()