from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris_data = load_iris()
irisDF = pd.DataFrame(data=iris_data.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
print(irisDF.info)

dbscan = DBSCAN(eps=0.6, min_samples=8, metric='euclidean')
dbscan_labels = dbscan.fit_predict(iris_data.data)
print("dbscan.core_sample_indices_: ", dbscan.core_sample_indices_)
print("dbscan.components_: ", dbscan.components_)
print("dbscan.labels_: ",dbscan.labels_)

irisDF['dbscan_cluster'] = dbscan_labels
irisDF['target'] = iris_data.target

iris_result = irisDF.groupby(['target'])['dbscan_cluster'].value_counts()
print(iris_result)

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris_data.data)

irisDF['pca_x'] = pca_transformed[:,0]
irisDF['pca_y'] = pca_transformed[:,1]
# print(irisDF)

maker0_ind = irisDF[irisDF['dbscan_cluster']==0].index
maker1_ind = irisDF[irisDF['dbscan_cluster']==1].index
maker2_ind = irisDF[irisDF['dbscan_cluster']==-1].index

plt.scatter(x=irisDF.loc[maker0_ind,'pca_x'], y=irisDF.loc[maker0_ind, 'pca_y'], marker='o')
plt.scatter(x=irisDF.loc[maker1_ind,'pca_x'], y=irisDF.loc[maker1_ind, 'pca_y'], marker='s')
plt.scatter(x=irisDF.loc[maker2_ind,'pca_x'], y=irisDF.loc[maker2_ind, 'pca_y'], marker='^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Cluster Visualization by 2 PCA Components')
plt.show()