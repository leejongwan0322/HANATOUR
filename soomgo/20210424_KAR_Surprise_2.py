import pandas as pd

# ratings = pd.read_csv(r'C:\Users\HANA\Downloads\ml-latest\ml-latest\ratings.csv')
# ratings.to_csv('C:\\Users\\HANA\\Downloads\\ml-latest\\ml-latest\\ratings_noh.csv', index=False, header=False)
from surprise import Reader, Dataset

# reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
# data = Dataset.load_from_file(r'C:\\Users\\HANA\\Downloads\\ml-latest\\ml-latest\\ratings_noh.csv', reader=reader)

from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy

# trainset, testset = train_test_split(data, test_size=.25, random_state=0)
# algo = SVD(n_factors=50, random_state=0)
# algo.fit(trainset)
# predictions = algo.test(testset)
# accuracy.rmse(predictions)

#cross_validate 실습
# from surprise.model_selection import cross_validate
ratings = pd.read_csv(r'C:\\Users\\HANA\\Downloads\\ml-latest-small\\ratings.csv')
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# algo = SVD(random_state=0)
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#GridSearchCV실습
from surprise.model_selection import GridSearchCV
param_grid = {'n_epochs': [20,40,60], 'n_factors': [50,100,200]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mse'], cv=3)
gs.fit(data)
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])