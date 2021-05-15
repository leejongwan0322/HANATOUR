import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# https://medium.com/analytics-vidhya/the-simplest-way-to-create-complex-visualizations-in-python-isnt-with-matplotlib-a5802f2dba92

data = pd.DataFrame(np.random.rand(10, 4), columns=['A','B','C','D'])
data.head()

# data.diff().plot.box(vert=False, color={'medians':'lightblue','boxes':'blue','caps':'darkblue'});
# plt.show()


data = pd.DataFrame(np.random.rand(100, 1), columns=['value']).reset_index()
# data['value'].plot()
# data['value'].rolling(10).mean().plot()
# plt.show()


# data = pd.DataFrame(np.random.rand(5, 2),index=list("ABCDE"), columns=list("XY"))
# data.plot.pie(subplots=True, figsize=(8, 4));
# plt.show()

data = pd.DataFrame(np.random.rand(100, 4), columns=['A','B','C','D'])
data.plot(subplots=True,figsize=(20,10));
plt.show()
