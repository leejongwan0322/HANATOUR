from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd

# news_data = fetch_20newsgroups(subset='all', random_state=156)
# print(news_data.keys())
# print('target 클래스의 값과 분포도\n', pd.Series(news_data.target).value_counts().sort_index())
# print('target Class''s feature name\n', news_data.target_names)
# print(news_data.DESCR)
# print(news_data.data[0])

train_news = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'),random_state=156)
X_train = train_news.data
# print(X_train)
y_train = train_news.target
# print(y_train)

test_news = fetch_20newsgroups(subset='test', remove=('headers','footers','quotes'),random_state=156)
X_test = test_news.data
y_test = test_news.target

print('test''s size {0}, train''s size {1}'.format(len(test_news.data), len(train_news.data)))

#Count Vectorization으로 train 데이타를 피처 벡터화 변환 수행
cnt_vect = CountVectorizer()
X_train_cnt_vect = cnt_vect.fit_transform(X_train)
 # = cnt_vect.transform(X_train)
print('CountVectorizer Train Size', X_train_cnt_vect.shape)
# print(X_train)
# print(X_train_cnt_vect)

#Train Data로 fit()된 ConVectorizer를 이용해 테스트 데이터를 Feature Vector화 변환 수행
X_test_cnt_vect = cnt_vect.transform(X_test)
print('CountVectorizer Test Size', X_test_cnt_vect.shape)

#LogisticRegression을 이용해 학습/예측/평가 수행
# lr_clf = LogisticRegression(solver='lbfgs', max_iter=100)
# lr_clf.fit(X_train_cnt_vect, y_train)
# pred = lr_clf.predict(X_test_cnt_vect)
# print('CountVectorized LogisticRegression Accuracy', accuracy_score(y_test, pred))

#TF-IDF 벡터화를 적용해 학습 데이터 세트와 테스트 데이터 세트 변환
# tfidf_vect = TfidfVectorizer()
tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300)
tfidf_vect.fit(X_train)
X_train_tfidf_vect = tfidf_vect.transform(X_train)
X_test_tfidf_vect = tfidf_vect.transform(X_test)

#LogisticRegression을 이용해 학습/예측/평가 수행
lr_clf = LogisticRegression()
# lr_clf.fit(X_train_tfidf_vect, y_train)
# pred = lr_clf.predict(X_test_tfidf_vect)
# print('CountVectorized LogisticRegression Accuracy', accuracy_score(y_test, pred))

#최적 C값 도출 튜닝 수행, CV 3 촐드 세트로 진행
params = {'C':[0.01,0.1,1,5,10]}
grid_cv_lr = GridSearchCV(lr_clf, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv_lr.fit(X_train_tfidf_vect, y_train)
print('LogisticRegression Best C Parameter :', grid_cv_lr.best_score_)

#최적 C값으로 학습된 grid_cv로 예측 및 정확도 평가
pred=grid_cv_lr.predict(X_test_tfidf_vect)
print('TF-IDF Vectorized LogisticRegression :', accuracy_score(y_test, pred))

pipline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300)),
    ('lr_clf', LogisticRegression(c=10))
])

pipline.fit(X_train, y_train)
pred = pipline.predict(X_test)
print(accuracy_score(y_test,pred))