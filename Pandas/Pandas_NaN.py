from numpy import NAN, NaN, nan

print(NaN == True)
print(NaN == False)
print(NaN == 0)
print(NaN =='')

print(NaN == NaN)
print(NaN == nan)
print(NaN == NAN)
print(nan == NAN)

import pandas as pd

print(pd.isnull(NaN))
print(pd.isnull(nan))
print(pd.isnull(NAN))

print(pd.notnull(NaN))
print(pd.notnull(42))
print(pd.notnull('missing'))

ebola = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\Pandas\doit_pandas-master\data\country_timeseries.csv')
print(ebola.value_counts())
# print(ebola.count())
# print(ebola.shape)
num_rows = ebola.shape[0]
num_missing = num_rows = ebola.count()
print(num_missing)


import numpy as np
print(np.count_nonzero(ebola.isnull()))
print(np.count_nonzero(ebola['Cases_Guinea'].isnull()))
print(ebola.Cases_Guinea.value_counts(dropna=False).head())

print(ebola.fillna(0).iloc[0:10, 0:5])
print(ebola.fillna(method='ffill').iloc[0:10, 0:5])
print(ebola.fillna(method='bfill').iloc[0:10, 0:5])
print(ebola.interpolate().iloc[0:10, 0:5])

print(ebola.shape)
ebola_dropna = ebola.dropna()
print(ebola_dropna.shape)
print(ebola_dropna.iloc[0:10, 0:5])

print(ebola.Cases_Guinea.sum(skipna=True))
print(ebola.Cases_Guinea.sum(skipna=False))