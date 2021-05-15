# https://alex-blog.tistory.com/entry/pythoncohort
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_columns', 9000)
#mpl.rcParams['lines.linewidth'] = 2

df = pd.read_excel('C:\\Users\\HANA\\\Downloads\\cohort_test.xlsx')
# print(df.head())

df['OrderPeriod'] = df.OrderDate.apply(lambda x: x.strftime('%Y-%m'))
# print(df.head())
# exit()

df.set_index('UserId', inplace=True)
df['CohortGroup'] = df.groupby(level=0)['OrderDate'].min().apply(lambda x: x.strftime('%Y-%m'))
df.reset_index(inplace=True)
# print(df.head())

# count the unique users, orders, and total revenue per Group + Period
grouped = df.groupby(['CohortGroup', 'OrderPeriod'])
cohorts = grouped.agg({'UserId': pd.Series.nunique, 'OrderId': pd.Series.nunique, 'TotalCharges': np.sum})
# print(df.head())
# exit()
# make the column names more meaningful
cohorts.rename(columns={'UserId': 'TotalUsers', 'OrderId': 'TotalOrders'}, inplace=True)
# print(cohorts.head())
# exit()


# def cohort_period(df):
#     """
#     Creates a `CohortPeriod` column, which is the Nth period based on the user's first purchase.
#
#     Example
#     -------
#     Say you want to get the 3rd month for every user:
#         df.sort(['UserId', 'OrderTime', inplace=True)
#         df = df.groupby('UserId').apply(cohort_period)
#         df[df.CohortPeriod == 3]
#     """

def cohort_period(df):
    print('---------')
    print(len(df))
    print(df)
    df['CohortPeriod'] = np.arange(len(df)) + 1
    print('---After------')
    print(df)
    return df



cohorts = cohorts.groupby(level=0).apply(cohort_period)
# cohorts.head()
# print('---------')
# print(cohorts)
# exit()


x = df[(df.CohortGroup == '2009-01') & (df.OrderPeriod == '2009-01')]
y = cohorts.loc[('2009-01', '2009-01')]

assert(x['UserId'].nunique() == y['TotalUsers'])
assert(x['TotalCharges'].sum().round(2) == y['TotalCharges'].round(2))
assert(x['OrderId'].nunique() == y['TotalOrders'])

x = df[(df.CohortGroup == '2009-01') & (df.OrderPeriod == '2009-09')]
y = cohorts.loc[('2009-01', '2009-09')]

assert(x['UserId'].nunique() == y['TotalUsers'])
assert(x['TotalCharges'].sum().round(2) == y['TotalCharges'].round(2))
assert(x['OrderId'].nunique() == y['TotalOrders'])

x = df[(df.CohortGroup == '2009-05') & (df.OrderPeriod == '2009-09')]
y = cohorts.loc[('2009-05', '2009-09')]

assert(x['UserId'].nunique() == y['TotalUsers'])
assert(x['TotalCharges'].sum().round(2) == y['TotalCharges'].round(2))
assert(x['OrderId'].nunique() == y['TotalOrders'])


# reindex the DataFrame
cohorts.reset_index(inplace=True)
cohorts.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)

# create a Series holding the total size of each CohortGroup
cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()
cohort_group_size.head()
cohorts['TotalUsers'].head()

print('-----Head()----------')
print(cohorts.head())
print(' ')
print('-----unstack()-------')
print(cohorts['TotalUsers'].unstack(0).head())
print(' ')
print('------divide()---------')
print(cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1).head())
print(' ')
# exit()

cohorts['TotalUsers'].unstack(0).head()
user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)
print(user_retention)

# user_retention[['2009-06', '2009-07', '2009-08']].plot(figsize=(10,5))
# plt.title('Cohorts: User Retention')
# plt.xticks(np.arange(1, 12.1, 1))
# plt.xlim(1, 12)
# plt.ylabel('% of Cohort Purchasing');
# plt.show()

import seaborn as sns
sns.set(style='white')


print('---------------------')
print(user_retention.T)
plt.figure(figsize=(12, 8))
plt.title('Cohorts: User Retention')
sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%');
plt.show()
