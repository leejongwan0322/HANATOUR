from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

import pandas as pd
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import Common_Module.CMStat as CM
import Common_Module.CMPlot as CMPlot

TOURDF = pd.read_excel(io='C:\\Users\\HANA\\Downloads\\ML.xlsx')
print("------------------------------------------------1\n")
print(TOURDF.info())
TOURDF = TOURDF[TOURDF['NTSL_AMT']>0]
print("------------------------------------------------2\n")
print(TOURDF.info())
# TOURDF = TOURDF[TOURDF['UnitPrice']>0]
print("------------------------------------------------3\n")
print(TOURDF.info())
TOURDF = TOURDF[TOURDF['CUST_NUM'].notnull()]
print("------------------------------------------------4\n")
print(TOURDF.shape)
print("------------------------------------------------5\n")
print(TOURDF.isnull().sum())
print("------------------------------------------------6\n")
print(TOURDF['AREA_CD'].value_counts()[:5])
print("------------------------------------------------7\n")
TOURDF = TOURDF[TOURDF['AREA_CD']=='EW']
TOURDF = TOURDF[TOURDF['RES_ATTR_CD']=='H']
print(TOURDF.shape)
print("------------------------------------------------8\n")
# TOURDF['sale_amount'] = TOURDF['Quantity'] * TOURDF['UnitPrice']
# TOURDF['CUST_NUM'] = TOURDF['CUST_NUM'].astype(int)
print(TOURDF['CUST_NUM'].value_counts().head())
print("------------------------------------------------9\n")
print(TOURDF.groupby('CUST_NUM')['NTSL_AMT'].sum().sort_values(ascending=False)[:5])
print("------------------------------------------------10\n")

aggregations = {
    'FRST_INP_DTTM': 'max',
    'CUST_NUM': 'count',
    'NTSL_AMT': 'sum'
}
cust_df = TOURDF.groupby('CUST_NUM').agg(aggregations)
cust_df = cust_df.rename(columns={'FRST_INP_DTTM':'Recency', 'CUST_NUM':'Frequency', 'NTSL_AMT':'Monetary'})
cust_df = cust_df.reset_index()
cust_df['Recency'] = datetime.datetime(2019,7,1) - cust_df['Recency']
cust_df['Recency'] = cust_df['Recency'].apply(lambda x:x.days+1)
print(cust_df)
print("------------------------------------------------11\n")


# fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12,4), nrows=1, ncols=3)
# ax1.set_title('Recency')
# ax1.hist(cust_df['Recency'])
#
# ax2.set_title('Frequency')
# ax2.hist(cust_df['Frequency'])
#
# ax3.set_title('Monetary')
# ax3.hist(cust_df['Monetary'])

# plt.show()


print(cust_df[['Recency','Frequency', 'Monetary']].describe())
print("------------------------------------------------12\n")
X_feature = cust_df[['Recency', 'Frequency', 'Monetary']].values
X_feature_scaled = StandardScaler().fit_transform(X_feature)
#
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict((X_feature_scaled))
cust_df['cluster_label'] = labels

print(silhouette_score(X_feature_scaled, labels))
print("------------------------------------------------13\n")
#
# CMPlot.visualize_silhouette([2,3,4,5], X_feature_scaled)
#
cust_df['Recency_log'] = np.log1p((cust_df['Recency']))
cust_df['Frequency_log'] = np.log1p((cust_df['Frequency']))
cust_df['Monetary_log'] = np.log1p((cust_df['Monetary']))
#
X_feature_log = cust_df[['Recency_log', 'Frequency_log', 'Monetary_log']].values
X_feature_scaled_log = StandardScaler().fit_transform(X_feature_log)
#
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict((X_feature_scaled_log))
cust_df['cluster_label_log'] = labels

print(silhouette_score(X_feature_scaled_log, labels))
print("------------------------------------------------14\n")
#
CMPlot.visualize_silhouette([2,3,4,5], X_feature_scaled_log)
