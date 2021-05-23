#StandardScaler
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

iris_set = load_iris()
iris_data = iris_set.data
irisDF = pd.DataFrame(data=iris_data, columns=iris_set.feature_names)
#
# print(irisDF.mean())
# print(irisDF.var())
#
# scaler = StandardScaler()
# scaler.fit(irisDF)
# iris_scaled = scaler.transform(irisDF)
#
# irisDF_scaled = pd.DataFrame(data=iris_scaled, columns=iris_set.feature_names)
#
# print(irisDF_scaled.mean())
# print(irisDF_scaled.var())


#MinMaxSclaer
# from sklearn.datasets import load_iris
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
#
# iris_set = load_iris()
# iris_data = iris_set.data
# irisDF = pd.DataFrame(data=iris_data, columns=iris_set.feature_names)
#
# scaler = MinMaxScaler()
# scaler.fit(irisDF)
# iris_scaled = scaler.transform(irisDF)
#
# #transform()시 스케일 변환된 데이터 세트가 Numpy ndarry로 반환돼 이를 DataFrame으로 변환
# irisDF_scaled = pd.DataFrame(data=iris_scaled, columns=iris_set.feature_names)
# print(irisDF_scaled.min())
# print(irisDF_scaled.max())


#학습 데이터와 테스트 데이터의 스케일링 변환 시 유의점

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import pandas as pd

iris_set = load_iris()
iris_data = iris_set.data
irisDF = pd.DataFrame(data=iris_data, columns=iris_set.feature_names)

train_array = np.arange(0,11).reshape(-1,1)
test_array = np.arange(0,6).reshape((-1,1))

scaler = MinMaxScaler()
scaler.fit(train_array)
train_caled = scaler.transform(train_array)
print("원본 train_array :", np.round(train_array.reshape(-1),2))
print("Scale된 train_array :", np.round(train_caled.reshape(-1),2))

# scaler.fit(test_array)
test_scaled = scaler.transform(test_array)
print("원본 test_array :", np.round(test_array.reshape(-1),2))
print("Scale된 train_array :", np.round(test_scaled.reshape(-1),2))