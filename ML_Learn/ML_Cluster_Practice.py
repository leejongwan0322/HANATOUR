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

retailDF = pd.read_excel(io='C:\\Users\\HANA\\PycharmProjects\\HANATOUR\\ML_Learn\\OnlineRetail.xlsx')
print(retailDF.info())
retailDF = retailDF[retailDF['Quantity']>0]
retailDF = retailDF[retailDF['UnitPrice']>0]
retailDF = retailDF[retailDF['CustomerID'].notnull()]
print(retailDF.shape)
print(retailDF.isnull().sum())
print(retailDF['Country'].value_counts()[:5])
retailDF = retailDF[retailDF['Country']=='United Kingdom']
print(retailDF.shape)
print('-------end')
retailDF['sale_amount'] = retailDF['Quantity'] * retailDF['UnitPrice']
retailDF['CustomerID'] = retailDF['CustomerID'].astype(int)
print(retailDF['CustomerID'].value_counts().head())
print(retailDF.groupby('CustomerID')['sale_amount'].sum().sort_values(ascending=False)[:5])

aggregations = {
    'InvoiceDate': 'max',
    'InvoiceNo': 'count',
    'sale_amount': 'sum'
}
cust_df = retailDF.groupby('CustomerID').agg(aggregations)
cust_df = cust_df.rename(columns={'InvoiceDate':'Recency', 'InvoiceNo':'Frequency', 'sale_amount':'Monetary'})
cust_df = cust_df.reset_index()
cust_df['Recency'] = datetime.datetime(2020,11,2) - cust_df['Recency']
cust_df['Recency'] = cust_df['Recency'].apply(lambda x:x.days+1)
print(cust_df)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12,4), nrows=1, ncols=3)
ax1.set_title('Recency')
ax1.hist(cust_df['Recency'])

ax2.set_title('Frequency')
ax2.hist(cust_df['Frequency'])

ax3.set_title('Monetary')
ax3.hist(cust_df['Monetary'])

# plt.show()

print(cust_df[['Recency','Frequency', 'Monetary']].describe())

X_feature = cust_df[['Recency', 'Frequency', 'Monetary']].values
X_feature_scaled = StandardScaler().fit_transform(X_feature)
#
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict((X_feature_scaled))
cust_df['cluster_label'] = labels
print(silhouette_score(X_feature_scaled, labels))
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
#
CMPlot.visualize_silhouette([2,3,4,5], X_feature_scaled_log)
# CMPlot.visualize_silhouette([2,3,4,5], X_feature_scaled_log)