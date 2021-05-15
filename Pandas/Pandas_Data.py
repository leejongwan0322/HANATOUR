import pandas as pd

pandas_DF = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\Pandas\doit_pandas-master\data\gapminder.tsv', sep='\t')
# print("\n-- type(pandas_DF)\n", type(pandas_DF))
# print("\n-- pandas_DF.columns\n", pandas_DF.columns)
# print("\n-- pandas_DF.shape\n", type(pandas_DF.shape))
# print("\n-- pandas_DF.shape\n", pandas_DF.shape)

country_df = pandas_DF['country']
# print("\n-- type(country_df)\n", type(country_df))
# print("\n-- country_df.head()\n", country_df.head())
# print("\n-- country_df.tail()\n", country_df.tail())

subset = pandas_DF[['country','continent','year']]
# print("\n-- type(subset)\n", type(subset))

# print("\n-- subset.head()\n", subset.head())
# print("\n-- subset.tail()\n", subset.tail())

print("\n-- pandas_DF.loc[0]\n", pandas_DF.loc[0])
print("\n-- pandas_DF.loc[99]\n", pandas_DF.loc[99])

number_of_rows = pandas_DF.shape[0]
last_row_index = number_of_rows - 1
# print("\n-- pandas_DF.loc[last_row_index]\n", pandas_DF.loc[last_row_index])
# print("\n-- pandas_DF.tail(1)\n", pandas_DF.tail(1))
#
print("\n-- pandas_DF.iloc[0]\n", pandas_DF.iloc[0])
print("\n-- pandas_DF.iloc[99]\n", pandas_DF.iloc[99])
# print("\n-- pandas_DF.iloc[-1]\n", pandas_DF.iloc[-1])
# print("\n-- pandas_DF.iloc[0,99,999]\n", pandas_DF.iloc[[0,99,999]])