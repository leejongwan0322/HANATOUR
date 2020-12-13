import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()
sclar = StandardScaler()
data_scled = sclar.fit_transform(cancer.data)

X_train, X_test, y_train, y_test = train_test_split(data_scled, cancer.target, test_size=0.3, random_state=0)

lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)

print('accuracy_score\n', accuracy_score(y_test, lr_pred))
print('roc_auc_score\n', roc_auc_score(y_test, lr_pred))

params={'penalty': ['l2'],
        'C': [0.01, 0.1, 1, 1, 5, 10]}
grid_clf = GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=3)
grid_clf.fit(data_scled, cancer.target)

print(grid_clf.best_params_, grid_clf.best_score_)