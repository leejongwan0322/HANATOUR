import Common_Module.CMStat as CM
from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)
bostonDF['PRICE'] = boston.target
y_target = bostonDF['PRICE']
x_data = bostonDF.drop(['PRICE'], axis=1, inplace=False)

lasso_alphas = [0.07,0.1,0.5,1,3]
coeff_lasso_df = CM.get_linear_reg_eval('Lasso', params=lasso_alphas, X_data_n=x_data, y_target_n=y_target)
print(coeff_lasso_df)

elastic_alphas = [0.07, 0.1, 0.5, 1, 3]
coeff_elastic_df = CM.get_linear_reg_eval('ElasticNet', params=elastic_alphas, X_data_n=x_data, y_target_n=y_target)
print(coeff_elastic_df)