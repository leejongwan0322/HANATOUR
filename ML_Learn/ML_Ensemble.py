import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
# print('cancer data set: ', cancer.data)
# print('cancer feature명: ', cancer.feature_names)
# print('cancer target명: ', cancer.target_names)
# print('cancer target값: ', cancer.target)
# print('cancer DESCRIPTION: ', cancer.DESCR)
# print('cancer frame: ', cancer.frame)
# keys = cancer.keys()
# print('cancer''s keys: ', cancer.keys())

data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
print(data_df)

lr_clf = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)
knn_clf = KNeighborsClassifier(n_neighbors=8)
rf_cf = RandomForestClassifier(random_state=0)

vo_clf = VotingClassifier(estimators=[('LR',lr_clf),('KNN',knn_clf),('RF',rf_cf)], voting='soft')
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)

vo_clf.fit(X_train, y_train)
pred = vo_clf.predict(X_test)
print("Voting 분류기 정확도", accuracy_score(y_test, pred))

classifiers = [lr_clf, knn_clf, rf_cf]
for classifier in classifiers:
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    class_name = classifier.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, accuracy_score(y_test,pred)))