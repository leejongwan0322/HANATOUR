import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading the data
data = pd.read_csv('https://www.openml.org/data/get_csv/1592290/phpgNaXZe')

# Setting up the column
column = ['sbp','tobacco','ldl','adiposity','famhist','type','obesity','alcohol','age','chd']
data.describe()
data.columns=column
print(data.head())
print(data.describe())

# Checking for any missing values
print(data.isnull().sum())


# Feature Scaling, making categorical data precise
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['famhist']=encoder.fit_transform(data['famhist'])
data['chd']=encoder.fit_transform(data['chd'])
print(data.head(5))


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range =(0,100))

# setting scale of max min value for sbp in range of 0-100, normalise
data['sbp'] = scale.fit_transform(data['sbp'].values.reshape(-1,1))

print(data.head())

# Data after modification
print(data.describe())

# data.head(50).plot(kind='area',figsize=(10,5))
# data.plot(x='age',y='obesity',kind='scatter',figsize =(10,5))
# data.plot(x='age',y='tobacco',kind='scatter',figsize =(10,5))
# data.plot(x='age',y='alcohol',kind='scatter',figsize =(10,5))
# data.plot(kind = 'hist',figsize =(10,5))
# color = dict(boxes='DarkGreen', whiskers='DarkOrange',medians='DarkBlue', caps='Gray')
# data.plot(kind='box',figsize=(10,6),color=color,ylim=[-10,90])
# plt.show()


# splitting the data into test and train  having a test size of 20% and 80% train size
from sklearn.model_selection import train_test_split
col = ['sbp','tobacco','ldl','adiposity','famhist','type','obesity','alcohol','age']
X_train, X_test, y_train, y_test = train_test_split(data[col], data['chd'], test_size=0.2, random_state=1234)


from sklearn import svm
svm_clf = svm.SVC(kernel ='linear')
svm_clf.fit(X_train,y_train)
y_pred_svm =svm_clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)

from sklearn.metrics import accuracy_score
svm_result = accuracy_score(y_test,y_pred_svm)
recall_svm = cm_svm[0][0]/(cm_svm[0][0] + cm_svm[0][1])
precision_svm = cm_svm[0][0]/(cm_svm[0][0]+cm_svm[1][1])
print("Accuracy :",svm_result)
print("Recall :",recall_svm)
print("Precision :",precision_svm)


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors =5,n_jobs = -1,leaf_size = 60,algorithm='brute')
knn_clf.fit(X_train,y_train)

y_pred_knn = knn_clf.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
knn_result = accuracy_score(y_test,y_pred_knn)
recall_knn = cm_knn[0][0]/(cm_knn[0][0] + cm_knn[0][1])
precision_knn = cm_knn[0][0]/(cm_knn[0][0]+cm_knn[1][1])
print("Accuracy :",knn_result)
print("Recall :",recall_knn)
print("Precision :",precision_knn)

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
ann_clf = MLPClassifier()

#Parameters
parameters = {'solver': ['lbfgs'],
             'alpha':[1e-4],
             'hidden_layer_sizes':(9,14,14,2),   # 9 input, 14-14 neuron in 2 layers,1 output layer
             'random_state': [1]}
# Type of scoring to compare parameter combos
acc_scorer = make_scorer(accuracy_score)

# Run grid search
grid_obj = GridSearchCV(ann_clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Pick the best combination of parameters
ann_clf = grid_obj.best_estimator_
# Fit the best algorithm to the data
ann_clf.fit(X_train, y_train)
y_pred_ann = ann_clf.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_ann = confusion_matrix(y_test, y_pred_ann)
ann_result = accuracy_score(y_test,y_pred_ann)
recall_ann = cm_ann[0][0]/(cm_ann[0][0] + cm_ann[0][1])
precision_ann = cm_ann[0][0]/(cm_ann[0][0]+cm_ann[1][1])
print("Accuracy :",ann_result)
print("Recall :",recall_ann)
print("Precision :",precision_ann)


results ={'Accuracy': [svm_result*100,knn_result*100,ann_result*100],
          'Recall': [recall_svm*100,recall_knn*100,recall_ann*100],
          'Precision': [precision_svm*100,precision_knn*100,precision_ann*100]}
index = ['SVM','KNN','ANN']
results =pd.DataFrame(results,index=index)
fig =results.plot(kind='bar',title='Comaprison of models',figsize =(9,9)).get_figure()
fig.savefig('Final Result.png')
plt.show()