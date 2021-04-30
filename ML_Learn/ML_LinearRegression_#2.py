from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)
bostonDF['PRICE'] = boston.target

y_target = bostonDF['PRICE']
x_data = bostonDF.drop(['PRICE'], axis=1, inplace=False)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_target, test_size=0.3, random_state=156)

#선형 회귀 OLS로 학습/예측/평가 수행
lr = LinearRegression()
lr.fit(X_train, y_train)
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)
print('\nMSE : {0:.3f}, RMSE : {1:.3F}\n'.format(mse, rmse))
print('\nr2 Variance score  {0:.3F}\n'.format(r2_score(y_test, y_preds)))

print('\n절편 값:\n', lr.intercept_)
print(boston.feature_names)
print('\n회귀 계수값:\n', lr.coef_)

# 회귀 계수를 큰 값 순으로 정렬하기 위해 Series로 생성, 인덱스 칼럼명에 유의
coeff = pd.Series(data=np.round(lr.coef_,1), index=x_data.columns)
print('\ncoeff.sort_values(ascending=False)\n', coeff.sort_values(ascending=False))

# cross_val_score()로 5폴드 세트로 MSE를 구한 뒤 이를 기반으로 다시 RMSE구함
neg_mse_scores = cross_val_score(lr, x_data, y_target, scoring="neg_mean_squared_error", cv=5)
print('\nneg_mse_scores\n', neg_mse_scores)
rmse_scores = np.sqrt(-1*neg_mse_scores)
print('\nrmse_scores\n', rmse_scores)
avg_rmse = np.mean(rmse_scores)

# neg_mean_squared_error로 반환된 값은 모두 음수
print('\n5 folds의 개별 Negative MSE scores: \n', np.round(neg_mse_scores,2))
print('\n5 folds의 개별 RMSE scores :', np.round((rmse_scores)))
print('\n5 folds의 평균 RMSE : {0:.3f}\n'.format(avg_rmse))