import pandas as pd
import sidetable as stb
import io
import requests
print(pd.get_option('display.max_rows'))
print(pd.get_option('display.max_colwidth'))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

DF_marketing = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\Pandas\doit_pandas-master\data\DirectMarketing.csv', sep=',')
print(DF_marketing.head())
#       Age  Gender OwnHome  Married  ... Children  History  Catalogs AmountSpent
# 0     Old  Female     Own   Single  ...        0     High         6         755
# 1  Middle    Male    Rent   Single  ...        0     High         6        1318
# 2   Young  Female    Rent   Single  ...        0      Low        18         296
# 3  Middle    Male     Own  Married  ...        1     High        18        2436
# 4  Middle  Female     Own   Single  ...        0     High        12        1304

print(DF_marketing.stb.freq(['Age']))
#       Age  count  percent  cumulative_count  cumulative_percent
# 0  Middle    508     50.8               508                50.8
# 1   Young    287     28.7               795                79.5
# 2     Old    205     20.5              1000               100.0

print(DF_marketing.stb.freq(['Age', 'Gender']))
#       Age  Gender  count  percent  cumulative_count  cumulative_percent
# 0  Middle    Male    302     30.2               302                30.2
# 1  Middle  Female    206     20.6               508                50.8
# 2   Young  Female    171     17.1               679                67.9
# 3     Old  Female    129     12.9               808                80.8
# 4   Young    Male    116     11.6               924                92.4
# 5     Old    Male     76      7.6              1000               100.0

print(DF_marketing.stb.freq(['Age'], value='AmountSpent'))
#       Age  AmountSpent    percent  cumulative_AmountSpent  cumulative_percent
# 0  Middle       762859  62.695415                  762859           62.695415
# 1     Old       293586  24.128307                 1056445           86.823722
# 2   Young       160325  13.176278                 1216770          100.000000


DF_election = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\Pandas\doit_pandas-master\data\governors_county.csv', sep=',')
print(DF_election.stb.freq(['state'], value='total_votes', thresh=40))
#             state  total_votes  ...  cumulative_total_votes  cumulative_percent
# 0  North Carolina      5525201  ...                 5525201           26.814545
# 1          others     15080038  ...                20605239          100.000000


print(DF_marketing.stb.counts())
#              count  unique  ... least_freq  least_freq_count
# Gender        1000       2  ...       Male               494
# OwnHome       1000       2  ...       Rent               484
# Married       1000       2  ...     Single               498
# Location      1000       2  ...        Far               290
# Age           1000       3  ...        Old               205
# History        697       3  ...     Medium               212
# Children      1000       4  ...          3               125
# Catalogs      1000       4  ...         24               233
# Salary        1000     636  ...      49000                 1
# AmountSpent   1000     852  ...        510                 1

print(DF_marketing.stb.counts(exclude='number'))
#           count  unique most_freq  most_freq_count least_freq  least_freq_count
# Gender     1000       2    Female              506       Male               494
# OwnHome    1000       2       Own              516       Rent               484
# Married    1000       2   Married              502     Single               498
# Location   1000       2     Close              710        Far               290
# Age        1000       3    Middle              508        Old               205
# History     697       3      High              255     Medium               212


print(DF_marketing.stb.missing())
#              missing  total  percent
# History          303   1000     30.3
# Age                0   1000      0.0
# Gender             0   1000      0.0
# OwnHome            0   1000      0.0
# Married            0   1000      0.0
# Location           0   1000      0.0
# Salary             0   1000      0.0
# Children           0   1000      0.0
# Catalogs           0   1000      0.0
# AmountSpent        0   1000      0.0


print(DF_marketing[['Age','OwnHome','AmountSpent']].groupby(['Age','OwnHome']).sum())
# Age    OwnHome   AmountSpent
# Middle Own           534259
#        Rent          228600
# Old    Own           230908
#        Rent           62678
# Young  Own            31091
#        Rent          129234

print(DF_marketing[['Age','OwnHome','AmountSpent']].groupby(['Age','OwnHome']).sum().stb.subtotal())
# Age         OwnHome            AmountSpent
# Middle      Own                     534259
#             Rent                    228600
#             Middle - subtotal       762859
# Old         Own                     230908
#             Rent                     62678
#             Old - subtotal          293586
# Young       Own                      31091
#             Rent                    129234
#             Young - subtotal        160325
# grand_total                        1216770

