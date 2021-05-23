from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#붓꽃 데이터 세트를 로딩합니다.
iris = load_iris()
# print('feature names''s shape:', len(iris.feature_names))
# print('feature names :', iris.feature_names)
#
# print('target names''s shape:', len(iris.target_names))
# print('target names :', iris.target_names)
#
# print('data shape :', iris.data.shape)
# print('data :', iris['data'])
#
# print('target shape :', iris.target.shape)
# print('target :', iris.target)

#iris.data는 Iris 데이터 세트에서 피처(feature)만으로 된 데이터를 numpy로 가지고 있습니다.
iris_data = iris.data


#iris.target은 붓꽃 데이터 세트에서 레이블(결정 값) 데이터를 numpy로 가지고 있습니다.
iris_label = iris.target
# print(iris_label)
# print(iris.target_names)

#붓꽃 데이터 세트를 자세히 보기 위해 dataframe으로 변환합니다.
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df.head(5))

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label,test_size=0.2, random_state=1)
print(X_train)
print(y_train)
dt_clf = DecisionTreeClassifier(random_state=11)

#학습수행
dt_clf.fit(X_train, y_train)

#학습이 완료된 DecisionTreeClassfier 객체에서 테스트 데이터 세트로 예측 수행.
pred = dt_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print('예측 정확도', accuracy_score(y_test, pred))


