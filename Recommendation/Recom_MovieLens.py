import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

movies = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\Recommendation\TEXT\movies.csv')
ratings = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\Recommendation\TEXT\ratings.csv')
# print(movies.shape)
# print(ratings.shape)

# print(movies.head(3))
# print(ratings.head(3))

ratings = ratings[['userId', 'movieId', 'rating']]
ratings_matrix = ratings.pivot_table('rating', index='userId', columns='movieId')
print(ratings_matrix.head(3))

rating_movies = pd.merge(ratings, movies, on='movieId')
ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')

ratings_matrix = ratings_matrix.fillna(0)
print(ratings_matrix.head(3))

rating_movies_T = ratings_matrix.transpose()
rating_movies_T.head(3)

from sklearn.metrics.pairwise import cosine_similarity
item_sim = cosine_similarity(rating_movies_T, rating_movies_T)

#cosine_similiarity()로 반환된 넘파이 행렬을 영화명을 매칭해 DataFame으로 변환
item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns, columns=ratings_matrix.columns)
print(item_sim_df.shape)
item_sim_df.head(3)