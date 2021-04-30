from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

mercari_df = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\NLP\TEXT_Example\mercari_train.tsv', sep='\t')
# print(mercari_df.shape)
# print(mercari_df.head(3))
print(mercari_df.info())

import matplotlib.pyplot as plt
import seaborn as sns

y_train_df = mercari_df['price']
# plt.figure(figsize=(6,4))
# sns.distplot(y_train_df, kde=False)
# plt.show()

import numpy as np

y_train_df = np.log1p(y_train_df)
# sns.distplot(y_train_df, kde=False)
# plt.show()
mercari_df['price'] = np.log1p(mercari_df['price'])
# print(mercari_df['price'].head(3))

# print(mercari_df['shipping'].value_counts())
# print(mercari_df['item_condition_id'].value_counts())
# boolean_cond = mercari_df['item_description']=='No description yet'
# print(mercari_df[boolean_cond]['item_description'].count())

def split_cat(category_name):
    try:
        return category_name.split('/')
    except:
        return ['Other_Null','Other_Null', 'Other_Null']

mercari_df['cat_dae'], mercari_df['cat_jung'], mercari_df['cat_so'] = \
    zip(*mercari_df['category_name'].apply(lambda x : split_cat(x)))
#
# print(mercari_df['cat_dae'].value_counts())
# print(mercari_df['cat_jung'].value_counts())
# print(mercari_df['cat_so'].value_counts())
#
# print(mercari_df['cat_jung'].nunique())
# print(mercari_df['cat_so'].nunique())

mercari_df['brand_name']= mercari_df['brand_name'].fillna(value='Other_Null')
mercari_df['category_name']= mercari_df['category_name'].fillna(value='Other_Null')
mercari_df['item_description']= mercari_df['item_description'].fillna(value='item_description')

# print(mercari_df.isnull().sum())

# print(mercari_df['brand_name'].nunique())
# print(mercari_df['brand_name'].value_counts()[:5])

# print(mercari_df['name'].nunique())
# print(mercari_df['name'].value_counts()[:10])

# print(mercari_df['item_description'].str.len().mean())
# print(mercari_df['item_description'][:3])

cnt_vec = CountVectorizer()
X_name = cnt_vec.fit_transform(mercari_df.name)

tfidf_descp = TfidfVectorizer(max_features=5000, ngram_range=(1,3), stop_words='english')
X_descp = tfidf_descp.fit_transform(mercari_df['item_description'])

print(X_name.shape)
print(X_descp.shape)

from sklearn.preprocessing import LabelBinarizer

#각 피처를 희소 행렬 원-핫 인코딩 변환
lb_brand_name = LabelBinarizer(sparse_output=True)
X_brand = lb_brand_name.fit_transform(mercari_df['brand_name'])

lb_item_cond_id = LabelBinarizer(sparse_output=True)
X_item_cond_id = lb_item_cond_id.fit_transform(mercari_df['item_condition_id'])

lb_shipping = LabelBinarizer(sparse_output=True)
X_shipping = lb_shipping.fit_transform(mercari_df['shipping'])

lb_cat_dae = LabelBinarizer(sparse_output=True)
X_cat_dae = lb_cat_dae.fit_transform(mercari_df['cat_dae'])

lb_cat_jung = LabelBinarizer(sparse_output=True)
X_cat_jung = lb_cat_jung.fit_transform(mercari_df['cat_jung'])

lb_cat_so = LabelBinarizer(sparse_output=True)
X_cat_so = lb_cat_so.fit_transform(mercari_df['cat_so'])

print(type(X_brand), type(X_item_cond_id))
print(X_brand.shape, X_item_cond_id.shape)

from scipy.sparse import hstack
import gc

sparse_matrix_list = (X_name, X_descp, X_brand, X_item_cond_id, X_shipping, X_cat_dae, X_cat_jung, X_cat_so)

#hstack 함수를 이용해 인코딩과 벡터화를 수행한데이터 세트를 모두 결합
X_features_sparse = hstack(sparse_matrix_list).tocsr()
print(type(X_features_sparse), X_features_sparse)

#데이터 세트가 메모리를 많이 차지하므로 사용 목적이 끝났으면 바로 메모리에서 삭제
del X_features_sparse
gc.collect()

def rmsle(y, y_pred):
    #underflow, overflow를 막기 위해 log가 아닌 log1p로 rmsle계산
    return np.sqrt(np.power(np.log1p(y)-np.log1p(y_pred), 2))

def evaluate_org_price(y_test, preds):
    #원본 데이터는 log1p로 변환되었으므로 exmpm1로 원복 필요
    preds_exmpm = np.expm1(preds)
    y_test_exmpm = np.expm1((y_test))

    #rmsle로 RMSLE값 추출
    rmsle_result = rmsle(y_test_exmpm, preds_exmpm)
    return rmsle_result

import gc
from scipy.sparse import hstack

def model_train_predict(model, matrix_ist):
    #scipy.sparse 모듈의 hstack을 이용해 희소 행렬 결합
    X = hstack(matrix_ist).tocsr()

    X_train, X_test, y_train, y_test = train_test_split(X, mercari_df['price'], test_size=0.2, random_state=156)

    #모델 학습 및 예측
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    del X, X_train, X_test, y_train
    gc.collect()

    return preds, y_test

linear_model = Ridge(solver='lsqr', fit_intercept=False)
sparse_matrix_list = (X_name, X_brand, X_item_cond_id, X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
linear_preds, y_test = model_train_predict(model=linear_model, matrix_ist=sparse_matrix_list)
print(evaluate_org_price(y_test, linear_preds))

sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id, X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
linear_preds, y_test = model_train_predict(model=linear_model, matrix_ist=sparse_matrix_list)
print(evaluate_org_price(y_test, linear_preds))



from lightgbm import LGBMRegressor
sparse_matrix_list = (X_descp, X_name, X_brand, X_item_cond_id, X_shipping, X_cat_dae, X_cat_jung, X_cat_so)
lgbm_model = LGBMRegressor(n_estimators=200, learning_rate=0.5, num_leaves=125, random_state=156)
lgbm_preds, y_test = model_train_predict(model=lgbm_model, matrix_ist=sparse_matrix_list)
print(evaluate_org_price(y_test, lgbm_preds))

preds = lgbm_preds*0.45+linear_preds*0.55
print(evaluate_org_price(preds))