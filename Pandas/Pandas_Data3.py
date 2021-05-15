import pandas as pd

# print("\n-- \n", "")
pandas_DF = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\Pandas\doit_pandas-master\data\gapminder.tsv', sep='\t')
print("\n-- pandas_DF.columns\n", pandas_DF.columns)

subset_loc = pandas_DF.loc[:, ['country','continent']]
print("\n-- subset loc\n", subset_loc)

subset_iloc = pandas_DF.iloc[:, [1,2,-1]]
print("\n-- subset iloc\n", subset_iloc)

subset_iloc_2 = pandas_DF.iloc[:, :4]
print("\n-- subset iloc #2\n", subset_iloc_2)

subset_iloc_3 = pandas_DF.iloc[:, 0:6:2]
print("\n-- subset iloc #3\n", subset_iloc_3)

subset_iloc_4 = pandas_DF.iloc[[0, 99, 999], 0:6:2]
print("\n-- subset iloc #4\n", subset_iloc_4)