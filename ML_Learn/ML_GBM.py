from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

cancer = load_breast_cancer()
data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
# print(data_df)

#GBM check timeing
start_time = time.time()
gb_clf = GradientBoostingClassifier(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)
gb_clf.fit(X_train, y_train)
pred = gb_clf.predict(X_test)
print(accuracy_score(y_test, pred))

param = {
    'n_estimators':[100,500,1000,2000,3000,10000],
    'learning_rate': [0.05,0.1]
}
grid_cv = GridSearchCV(gb_clf, param_grid=param, cv=2, verbose=1)
grid_cv.fit(X_train, y_train)
print('Best Highper parameter: ', grid_cv.best_params_)
print('Optimun accuracy rate: ', format(grid_cv.best_score_))

gb_pred = grid_cv.best_estimator_.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print('GBM optimun: ', gb_accuracy)