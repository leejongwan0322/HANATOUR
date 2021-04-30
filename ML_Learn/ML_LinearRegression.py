import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.DESCR)
bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)
bostonDF['PRICE'] = boston.target

print("#bostonDF\n", bostonDF)
print("#bostonDF.info()\n", bostonDF.info())
print("#bostonDF.describe()\n", bostonDF.describe())

fig, axs = plt.subplots(figsize=(16,8), ncols=5, nrows=2)
lm_features = ['ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'RAD','PTRATIO', 'LSTAT','CHAS','B']

for i, feature in enumerate(lm_features):
    row = int(i//5)
    col = i%5
    sns.regplot(x=feature, y='PRICE', data=bostonDF, ax=axs[row][col])
plt.show()
