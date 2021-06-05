from sklearn.datasets import load_iris, make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

#Iris Data
iris_data = load_iris()
# iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
dt_clf = DecisionTreeClassifier(random_state=11)
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=11, shuffle=True)
model_all_params = dt_clf.fit(iris_data.data, iris_data.target)
model_all_params = dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print(accuracy_score(y_test,pred))

#feature importance 추출
print(dt_clf.feature_importances_)

#feature별 importance 매핑
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
    print(name,value)

sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)
plt.show()