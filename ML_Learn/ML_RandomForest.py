from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

cancer = load_breast_cancer()
data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
# print(data_df)

rf_clf = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
print(accuracy_score(y_test, pred))

params = {
    'n_estimators':[100],
    'max_depth': [6,8,10,12],
    'min_samples_leaf': [8,12,18],
    'min_samples_split': [8,16,20]
}
rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
print(accuracy_score(y_test, pred))


#Grid Search
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
grid_cv.fit(X_train,y_train)
print('Best Parameter', grid_cv.best_params_)
print('Best Accuracy {0:.4f}'.format(grid_cv.best_score_))

rf_clf1 = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=8, min_samples_split=8, random_state=0)
rf_clf1.fit(X_train, y_train)
pred=rf_clf1.predict(X_test)
print('Accuracy {0:.4f}'.format(accuracy_score(y_test,pred)))
ftr_importances_values = rf_clf1.feature_importances_
ftr_importances = pd.Series(ftr_importances_values)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
print(ftr_top20)

plt.figure(figsize=(8,6))
plt.title('Figure importances Top 20')
sns.barplot(x=ftr_top20.index, y=ftr_top20)
plt.show()