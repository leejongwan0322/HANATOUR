import pandas as pd
print(pd.get_option('display.max_rows'))
print(pd.get_option('display.max_colwidth'))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', 800)


pandas_DF = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\Pandas\doit_pandas-master\data\gapminder.tsv', sep='\t')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
print(pandas_DF)

# print(type(pandas_DF))
#
# print(pandas_DF.head(10))
# print(pandas_DF.shape)
#
# print(pandas_DF.info())
print(pandas_DF.describe())
#
# print(pandas_DF['year'].value_counts())

year_feature = pandas_DF['year']
# print(year_feature.head(10))
year_value = pandas_DF['year'].value_counts()
# print(year_value)

# print(pandas_DF.columns)
