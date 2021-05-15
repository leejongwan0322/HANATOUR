import pandas as pd

# print("\n-- \n", "")
pandas_DF = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\Pandas\doit_pandas-master\data\gapminder.tsv', sep='\t')
print("\n-- pandas_DF.columns\n", pandas_DF.columns)

subset = pandas_DF.loc[:, ['year', 'pop']]
print("\n-- loc subset\n", subset)

subset = pandas_DF.iloc[:, [2, 4, -1]]
print("\n-- iloc subset\n", subset)

range_1 = range(10)
print("\n-- list(range_1)\n", list(range_1))
print("\n-- tuple(range_1)\n", tuple(range_1))

range_2 = range(1, 10)
print("\n-- list(range_2)\n", list(range_2))

range_3 = range(0, 10, 2)
print("\n-- list(range_3)\n", list(range_3))

range_4 = range(10, 1, -2)
print("\n-- list(range_4)\n", list(range_4))

for i in range_4:
    print(i)

for i in enumerate(range_4):
    print(i)

for i, v in enumerate(range_4):
    print(i, v)