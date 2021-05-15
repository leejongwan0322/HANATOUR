# grid_parameters = {'max_depth': [1,2,3], 'min_samples_split': [2,3]}

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

iris = load_iris()
iris_data = iris.data
x_train, X_test, y_train, y_test\
    = train_test_split(iris.data, iris.target, test_size=0.2, random_state=121)
dtree = DecisionTreeClassifier()

parameters = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)
grid_dtree.fit(x_train, y_train)
score_df = pd.DataFrame(grid_dtree.cv_results_)
print(score_df[['params','mean_test_score','rank_test_score']])

print('최적 파라미터:', grid_dtree.best_params_)
print('최고 정확도:', grid_dtree.best_score_)

#GridSearchCV의 refit으로 이미 학습된 estimator 반환
estimator = grid_dtree.best_estimator_
pred = estimator.predict(X_test)

#GridSearchCV의 best_estimator_는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
from sklearn.metrics import accuracy_score
print('예측 정확도', accuracy_score(y_test, pred))