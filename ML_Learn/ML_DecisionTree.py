from sklearn.datasets import load_iris, make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
# import ML_Learn.Common_Module as Common_Module

#Iris Data
iris_data = load_iris()
# print(iris_data)
# print('iris data set: ', iris_data.data)
# print('iris feature명: ', iris_data.feature_names)
# print('iris target명: ', iris_data.target_names)
# print('iris target값: ', iris_data.target)
# print('iris DESCRIPTION: ', iris_data.DESCR)
# print('iris frame: ', iris_data.frame)
# keys = iris_data.keys()
# print('iris''s keys: ', iris_data.keys())

iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
# print(iris_df)
# print(iris_df.describe())
# iris_df['label'] = iris_data.target
# print(iris_df.describe())
# print(iris_df.info)
#DecisionTree Classifier Creat
dt_clf = DecisionTreeClassifier(random_state=11)
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=11, shuffle=True)

model_all_params = dt_clf.fit(iris_data.data, iris_data.target)

model_all_params = dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print(accuracy_score(y_test,pred))

# Prepare a plot figure with set size.
plt.figure(figsize=(8,8))
# Plot the decision tree, showing the decisive values and the improvements in Gini impurity along the way.
plot_tree(model_all_params, filled=True, impurity=True)
# Display the tree plot figure.
plt.show()

# print('Featuew importance:\n{0}'.format(np.round(dt_clf.feature_importances_,3)))
# for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
#     print('{0}:{1:.3f}'.format(name, value))

# sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)
# plt.show()

# plt.title("3 Class values with 2 Feature Sample Data Creation")
# X_features, y_labels = make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes=3, n_clusters_per_class=1, random_state=0)
# plt.scatter(X_features[:,0], X_features[:,1],marker='o',c=y_labels,s=25,edgecolors='k')

# dt_clf = DecisionTreeClassifier().fit(X_features, y_labels)
# Common_Module.visualize_boundary(dt_clf,X_features,y_labels)

# dt_clf2 = DecisionTreeClassifier(min_samples_leaf=6).fit(X_features, y_labels)
# Common_Module.visualize_boundary(dt_clf2,X_features,y_labels)
# plt.show()
