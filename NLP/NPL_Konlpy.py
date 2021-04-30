import pandas as pd
import re

train_df = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\NLP\TEXT_Example\naver\ratings_train.txt', sep='\t')
print(train_df.head())
print(train_df['label'].value_counts())
train_df = train_df.fillna(' ')
train_df['document'] = train_df['document'].apply(lambda x : re.sub(r"\d+"," ", x))

test_df = pd.read_csv(r'C:\Users\HANA\PycharmProjects\HANATOUR\NLP\TEXT_Example\naver\ratings_test.txt', sep='\t')
print(test_df.head())
print(test_df['label'].value_counts())
test_df = test_df.fillna(' ')
test_df['document'] = test_df['document'].apply(lambda x : re.sub(r"\d+"," ", x))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import Common_Module.CMNLP as CMNLP

tfidf_vect = TfidfVectorizer(tokenizer=CMNLP.tw_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
tfidf_vect.fit(train_df['document'])
tfidf_matrix_train = tfidf_vect.transform(train_df['document'])

#LogisticRegression를 이용하여 감성 분석 분류 수행.
lg_clf = LogisticRegression(random_state=0)

params = {'C':[1,2]}
grid_cv = GridSearchCV(lg_clf, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv.fit(tfidf_matrix_train, train_df['label'])
print(grid_cv.best_params_, grid_cv.best_score_)

from sklearn.metrics import accuracy_score

#학습 데이터를 적용한 TfidfVectorizer를 이용하여 테스트 데이터를 TF-IDF값으로 피처 변환함
tfidf_matrix_test = tfidf_vect.transform(test_df['document'])

#Classifier는 GridSearchCV에서 최적 파라미터로 학습된 Classifier를 그대로 이용
best_estimator = grid_cv.best_estimator_
preds = best_estimator.predict(tfidf_matrix_test)

print(accuracy_score(test_df['label'], preds))