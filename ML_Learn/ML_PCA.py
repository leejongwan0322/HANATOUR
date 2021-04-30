from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
columns = ['sepal_length','sepal_width','petal_length','petal_width']
irisDF = pd.DataFrame(iris.data, columns=columns)
irisDF['target']=iris.target
print(irisDF)


# markers=['^','s','o']
# for i, marker in enumerate(markers):
#     x_axis_data = irisDF[irisDF['target']==i]['sepal_length']
#     y_axis_data = irisDF[irisDF['target']==i]['sepal_width']
#     plt.scatter(x_axis_data, y_axis_data, marker=marker, label=iris.target_names[i])
#
# plt.legend()
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.show()

iris_scaled = StandardScaler().fit_transform(irisDF.iloc[:, :-1])

pca = PCA(n_components=2)
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)
print(iris_pca.shape)

pca_columns=['pca_component_1', 'pca_component_2']
irisDF_pca = pd.DataFrame(iris_pca, columns=pca_columns)
irisDF_pca['target'] = iris.target
print(irisDF_pca)


# markers=['^','s','o']
# for i, marker in enumerate(markers):
#     x_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_1']
#     y_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_2']
#     plt.scatter(x_axis_data, y_axis_data, marker=marker, label=iris.target_names[i])

# plt.legend()
# plt.xlabel('pca_component_1')
# plt.ylabel('pca_component_2')
# plt.show()

print(pca.explained_variance_)

rcf = RandomForestClassifier(random_state=156)
scores = cross_val_score(rcf, iris.data, iris.target, scoring='accuracy', cv=3)
print('원본 데이터 교차 검증 개별 정확도', scores)
print('원본 데이터 평균 정확도', np.mean(scores))

pca_X = irisDF_pca[['pca_component_1','pca_component_2']]
scores_pca = cross_val_score(rcf, pca_X, iris.target, scoring='accuracy', cv=3)
print('PCA 원본 데이터 교차 검증 개별 정확도', scores_pca)
print('PCA 원본 데이터 평균 정확도', np.mean(scores_pca))
