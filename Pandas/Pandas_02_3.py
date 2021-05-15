import pandas as pd
import matplotlib.pyplot as plt


# print("\n-- \n", "")
pandas_DF = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\Pandas\doit_pandas-master\data\gapminder.tsv', sep='\t')
print("\n-- pandas_DF.columns\n", pandas_DF.columns)

#0~9개만 나오게 한다.
print(pandas_DF.head(n=10))

#select mean(lifeExp) from pandas_DF gorup by year
print(pandas_DF.groupby('year')['lifeExp'].mean())

grouped_year_df = pandas_DF.groupby('year')
print('grouped_year_df\n', grouped_year_df)

grouped_year_df_lifeExp = grouped_year_df['lifeExp']
mean_lifeExp_by_year = grouped_year_df_lifeExp.mean()
print('mean_lifeExp_by_year\n', mean_lifeExp_by_year)

multi_group_var = pandas_DF.groupby(['year','continent'])[['lifeExp', 'gdpPercap']].mean()
print('multi_group_var\n', multi_group_var)

print('nonique\n', pandas_DF.groupby('continent')['country'].nunique())

#그래프그리기
pandas_DF.groupby('year')['lifeExp'].mean().plot()
plt.show()