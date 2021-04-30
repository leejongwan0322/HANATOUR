import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import Common_Module.CMStat as CMStat
import Common_Module.CMPlot as CMPlot
import numpy as np

titanic_df = pd.read_csv('C:\\Users\\HANA\\PycharmProjects\\HANATOUR\\Pandas\\doit_pandas-master\\data\\train.csv')
# print(titanic_df.info())
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis=1)
X_titanic_df = CMStat.transform_features(X_titanic_df)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)
# print(X_titanic_df)

lr_clf = LogisticRegression(max_iter=4000)
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba_class1 =lr_clf.predict_proba(X_test)[:,1]

fprs, tprs, threadholds = roc_curve(y_test, pred_proba_class1)
# print(fprs)
thr_index = np.arange(1, threadholds.shape[0],5)

# print('Sample 추출을 위한 임계값 배열의 index 10개:', thr_index)
# print('Sample 10개의 임계값: ', np.round(threadholds[thr_index], 2))
# print('Sample 임계값 FPR: ', np.round(fprs[thr_index], 3))
# print('Sample 임계값 TPR: ', np.round(tprs[thr_index], 3))

# print(confusion_matrix(y_test, pred))
# print("정확도: ", accuracy_score(y_test, pred))
# print("정밀도: ", precision_score(y_test, pred))
# print("재현율: ", recall_score(y_test, pred))
# print(f1_score(y_test, pred))

CMStat.get_clf_eval(y_test, pred)
CMPlot.roc_curve_plot(y_test, pred_proba_class1)