import pandas as pd
import re

review_df = pd.read_csv('C:\\Users\\HANA\\PycharmProjects\\HANATOUR\\NLP\\TEXT_Example\\labeledTrainData.tsv', header=0, sep="\t", quoting=3)

print(review_df.head())
# print(review_df['review'][0])

#불필요한 내용 정리 하기에서 위와 아래는 동일한 내용
review_df['review'] = review_df['review'].str.replace('<br />', ' ')
review_df['review'] = review_df['review'].apply(lambda x : re.sub("[^a-zA-Z]"," ", x))

from sklearn.model_selection import train_test_split

class_df = review_df['sentiment']
print(class_df)

feature_df = review_df.drop(['id','sentiment'], axis=1, inplace=False)
print(feature_df)

X_train, X_test, y_train, y_test = train_test_split(feature_df, class_df, test_size=0.3, random_state=156)
print(X_train.shape, X_test.shape)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

#스톱 워드는 English, Filtering, ngram은 (1,2)로 설정해 CountVectorizer 수행
#LogisticRegression의 C는 10으로 설정
pipline = Pipeline([
    ('cnt_vect', CountVectorizer(stop_words='english', ngram_range=(1,2))),
    ('lt_clf', LogisticRegression(C=10))])

#Pipline 객체를 이용해 fit(), prefit()로 학습/예측 수행, predict_proba()는 roc_auc때문에 수행
pipline.fit(X_train['review'], y_train)
pred = pipline.predict(X_test['review'])
pred_prods = pipline.predict_proba(X_test['review'])[:, 1]
print(accuracy_score(y_test, pred), roc_auc_score(y_test, pred_prods))


#스톱 워드는 English, Filtering, ngram은 (1,2)로 설정해 TfidfVectorizer 수행
#LogisticRegression의 C는 10으로 설정
pipline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('lt_clf', LogisticRegression(C=10))])

pipline.fit(X_train['review'], y_train)
pred = pipline.predict(X_test['review'])
pred_prods = pipline.predict_proba(X_test['review'])[:, 1]
print(accuracy_score(y_test, pred), roc_auc_score(y_test, pred_prods))