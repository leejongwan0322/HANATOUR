from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

#Data가 적재되었다는 가정으로. 여기서는 Iris를 사용합니다.
iris_data = load_iris()
irisDF = pd.DataFrame(data=iris_data.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
print(irisDF.head())

#Coulmn 추가 방법 #1
irisDF['new_column'] = ['Large' if t else 'Small' for t in list(irisDF['sepal_length'] > 4)]
print(irisDF.head())

#Coulmn 추가 방법 #2
dict_ = {True : "Large", False : "Small"}
irisDF["new_columns"] = [dict_[t] for t in list(irisDF["sepal_length"] > 5)]
print(irisDF.head())

#Coulmn 추가 방법 #3
irisDF["new_columns"] = np.where(irisDF["sepal_length"].values > 5, "Large", "Small")
print(irisDF.head())

#Coulmn 추가 방법 #4
def func(x):
    if x > 5:
        return "Large"
    else:
        return "Small"

irisDF["new_columns"] = irisDF["sepal_length"].apply(lambda x : func(x))
print(irisDF.head())