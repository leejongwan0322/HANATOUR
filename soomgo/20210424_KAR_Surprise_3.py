import pandas as pd
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
from surprise.dataset import DatasetAutoFolds

ratings = pd.read_csv(r'C:\\Users\\HANA\\Downloads\\ml-latest-small\\ml-latest-small\\ratings.csv')
ratings.to_csv('C:\\Users\\HANA\\Downloads\\ml-latest-small\\ml-latest-small\\ratings_noh.csv', index=False, header=False)

reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0, 0.5))
data_folds = DatasetAutoFolds(ratings_file='C:\\Users\\HANA\\Downloads\\ml-latest-small\\ml-latest-small\\ratings_noh.csv', reader=reader)
trainset = data_folds.build_full_trainset()
algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)

movies = pd.read_csv(r'C:\\Users\\HANA\\Downloads\\ml-latest-small\\ml-latest-small\\movies.csv')
movieIds = ratings[ratings['userId']==9]['movieId']
# print(movieid)

if movieIds[movieIds==42].count() == 0:
    print('There is no rating of 42 for user 9')

# print(movies[movies['movieId']] == 42)

uid = str(9)
iid = str(42)

pred = algo.predict(uid, iid, verbose=True)

def get_unseen_surprise(ratings, movies, userId):
    seen_movies = ratings[ratings['userId'] == userId]['movieId'].tolist()
    total_movies = movies['movieId'].tolist()
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]
    print('평점 매긴 영화 수:', len(seen_movies), '추천 대상 영화 수:', len(unseen_movies), '전체 영화 수:', len(total_movies))

    return unseen_movies

unseen_movies = get_unseen_surprise(ratings, movies, 9)
# print(unseen_movies)

def recomm_movie_by_surprise(algo, userId, unseen_movies, top_n=10):

    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]

    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]
    top_movie_ids = [int(pred.iid) for pred in top_predictions]
    top_movie_rating = [pred.est for pred in top_predictions]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)]['title']
    top_movie_pred = [(id, title, rating) for id, title, rating in zip(top_movie_ids, top_movie_titles, top_movie_rating)]

    return top_movie_pred

unseen_movies = get_unseen_surprise(ratings, movies, 9)
top_movie_preds = recomm_movie_by_surprise(algo, 9, unseen_movies, top_n=10)

print('### Recommendation Top 10 #####')
for top_movie in top_movie_preds:
    print(top_movie[1], ":", top_movie[2])