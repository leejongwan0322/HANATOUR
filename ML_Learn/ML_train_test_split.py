from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

iris = load_iris()
iris_data = iris.data
iris_label = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=15)
print(X_train)
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))

scores = cross_val_score(dt_clf, iris_data, iris_label, scoring='accuracy', cv=3)
print('교차 검증별 정확도\n', np.round(scores,4))
print('평균 검증 정확도\n', np.round(np.mean(scores),4))