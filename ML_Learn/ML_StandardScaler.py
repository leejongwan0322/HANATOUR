from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

iris_set = load_iris()
iris_data = iris_set.data
irisDF = pd.DataFrame(data=iris_data, columns=iris_set.feature_names)

print(irisDF.mean())
print(irisDF.var())

scaler = StandardScaler()
scaler.fit(irisDF)
iris_scaled = scaler.transform(irisDF)

irisDF_scaled = pd.DataFrame(data=iris_scaled, columns=iris_set.feature_names)

print(irisDF_scaled.mean())
print(irisDF_scaled.var())
