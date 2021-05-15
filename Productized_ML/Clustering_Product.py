from sklearn.cluster import KMeans
import numpy as np
import os
import pandas as pd

dir_location = 'D:\\Data관련\\'
file_location = 'cluster_data.xlsx'
excel_file_dir = os.path.join(dir_location, file_location)

df_from_excel = pd.read_excel(excel_file_dir,
                              sheet_name='data',
                              header=0,
                              # index_col='SALE_PROD_CD',
                              thousands=',',
                              nrows=10,
                              comment='#',
                              dtype={
                                'PROD_MSTR_CD':str,
                                'RPRS_PROD_CD':str,
                                'SALE_PROD_CD':str,
                                'S_AREACODE':str,
                                'DEP_DT':str,
                                'DEP_TM':str,
                                'PROD_AREA_CD':str,
                                'PROD_MSTR_NM':str,
                                'PROD_BRND_NM':str,
                                'SALE_PROD_NM':str,
                                'PROD_DTL_ATTR_CD':str,
                                'AMT_AVG':str,
                                'CITY_CD':str,
                                'CITY_NM':str,
                                'CNTRY_NM':str,
                                'CNTRY_CD':str,
                                'RPRS_PROD_CNTNT_URL_ADRS':str,
                                'TOTAL':float
                              }
                              ).dropna()

df_from_excel['DEP_DATE'] = pd.to_datetime(df_from_excel['DEP_DT'] + df_from_excel['DEP_TM'], format='%Y%m%d%H%M')
df_from_excel['AMT_AVG'] = pd.to_numeric(df_from_excel['AMT_AVG'])
df_from_excel['DEP_DATE'] = pd.to_datetime(df_from_excel['DEP_DATE'], format='%Y%m%d%H%M')
print(df_from_excel.head())
print(df_from_excel.info())
print(df_from_excel.loc[:,['AMT_AVG','TOTAL']])

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(df_from_excel.loc[:,['AMT_AVG','TOTAL']])
print(kmeans.labels_)
df_from_excel['cluster'] = kmeans.labels_
