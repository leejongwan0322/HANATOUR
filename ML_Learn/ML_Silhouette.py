from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris_data = load_iris()

irisDF = pd.DataFrame(data=iris_data.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(irisDF)
irisDF['cluster'] = kmeans.labels_

score_samples = silhouette_samples(iris_data.data, irisDF['cluster'])
# print(score_samples.shape)
print(score_samples)
irisDF['silhoutte_coeff'] = score_samples

average_score = silhouette_score(iris_data.data, irisDF['cluster'])
print(average_score)
print(irisDF.groupby('cluster')['silhoutte_coeff'].mean())