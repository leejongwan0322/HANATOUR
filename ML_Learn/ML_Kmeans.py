from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import Common_Module.CMPlot as CMPlot
import Common_Module.CMStat as CMStat
import pandas as pd
import time

iris_data = load_iris()
print('iris data set: ', iris_data.data)
print('iris feature명: ', iris_data.feature_names)
# print('iris target명: ', iris_data.target_names)
# print('iris target값: ', iris_data.target)
# print('iris DESCRIPTION: ', iris_data.DESCR)
# print('iris frame: ', iris_data.frame)
# keys = iris_data.keys()
# print('iris''s keys: ', iris_data.keys())

irisDF = pd.DataFrame(data=iris_data.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
# print(irisDF)
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(irisDF)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# print(kmeans.n_iter_)
# print(kmeans.inertia_)

irisDF['target'] = iris_data.target
irisDF['cluster'] = kmeans.labels_
# print(irisDF)
iris_result = irisDF.groupby(['target','cluster'])['sepal_length'].count()
print(iris_result)

# CMPlot.visualize_elbowmethod(irisDF)
# CMPlot.visualize_silhouette_layer(irisDF)

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris_data.data)

irisDF['pca_x'] = pca_transformed[:,0]
irisDF['pca_y'] = pca_transformed[:,1]
# print(irisDF)

maker0_ind = irisDF[irisDF['cluster']==0].index
maker1_ind = irisDF[irisDF['cluster']==1].index
maker2_ind = irisDF[irisDF['cluster']==2].index
maker3_ind = irisDF[irisDF['cluster']==3].index

plt.scatter(x=irisDF.loc[maker0_ind,'pca_x'], y=irisDF.loc[maker0_ind, 'pca_y'], marker='o')
plt.scatter(x=irisDF.loc[maker1_ind,'pca_x'], y=irisDF.loc[maker1_ind, 'pca_y'], marker='s')
plt.scatter(x=irisDF.loc[maker2_ind,'pca_x'], y=irisDF.loc[maker2_ind, 'pca_y'], marker='^')
plt.scatter(x=irisDF.loc[maker3_ind,'pca_x'], y=irisDF.loc[maker3_ind, 'pca_y'], marker='^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Cluster Visualization by 2 PCA Components')
print(time.process_time())
plt.show()

