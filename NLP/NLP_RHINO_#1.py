import os

def read_data(filename):
    with open(filename, 'r', encoding="cp949") as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

data = read_data(r'C:\Users\HANA\PycharmProjects\HANATOUR\NLP\TEXT_Example\ratings_morphed.txt')
data_text = [line[1] for line in data]
data_senti = [line[2] for line in data]

from sklearn.model_selection import train_test_split

train_data_text, test_data_text, train_data_senti, test_data_senti \
    = train_test_split(
        data_text,
        data_senti,
        stratify=data_senti,
        test_size=0.3,
        random_state=156
)

#Test와 Train Data Set의 수를 확인해봅니다. 단순한 체크
from collections import Counter
train_data_senti_freq = Counter(train_data_senti)
print('train_data_senti_freq:', train_data_senti_freq)
test_data_senti_freq = Counter(test_data_senti)
print('test_data_senti_freq:', test_data_senti_freq)

#리뷰계산을 위해서 벡터화
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(train_data_text)
X_train = vect.transform(train_data_text)

print("X_train:\n", repr(X_train))
feature_names = vect.get_feature_names()
print("특성 개수:", len(feature_names))
print("처음 20개 특성:\m", feature_names[:20])
print("3000~5000까지의 특성:\n", feature_names[3000:5000])

#리뷰계산을 위해서 벡터화 테스트삼아 TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=5, ngram_range=(1,2)).fit(train_data_text)
X_train = vect.transform(train_data_text)

print("X_train:\n", repr(X_train))
feature_names = vect.get_feature_names()
print("특성 개수:", len(feature_names))
print("처음 20개 특성:\m", feature_names[:20])
print("3000~5000까지의 특성:\n", feature_names[3000:5000])

#벡터화한 데이터를 학습하고 다시 검증하고..
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
y_train = pd.Series(train_data_senti)
scores = cross_val_score(LogisticRegression(solver="liblinear", C=1), X_train, y_train, cv=5)
print('교차 검증 점수:', scores)
print('교차 검증 점수 평균:', scores.mean())

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 3, 5]}
grid = GridSearchCV(LogisticRegression(solver="liblinear"), param_grid, cv=5)
grid.fit(X_train, y_train)
print("최고 교차 검증 점수:", round(grid.best_score_, 3))
print("최적의 매개변수:", grid.best_params_)

X_test = vect.transform(test_data_text)
y_test = pd.Series(test_data_senti)
print("테스트 데이터 점수:", grid.score(X_test, y_test))

#위에서 검증하고 학습한 모델을 가지고.. 진짜로 다른 데이타를 넣어보는거
import rhinoMorph
rn = rhinoMorph.startRhino()
print('rn\n',rn)
new_input = '오늘은 정말 재미있는 하루구나!'

# 입력 데이터 형태소 분석하기
inputdata = []
morphed_input = rhinoMorph.onlyMorph_list(rn, new_input, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'])
morphed_input = ' '.join(morphed_input)                     # 한 개의 문자열로 만들기
inputdata.append(morphed_input)                               # 분석 결과를 리스트로 만들기
X_input = vect.transform(inputdata)
print(float(grid.predict(X_input)))
result = grid.predict(X_input) # 0은 부정,1은 긍정
print(result)
print(type(result))
if result == 0.0:
    print("부정적인 글입니다")
else:
    print("긍정적인 글입니다")


new_input = '식당은 두고두고 생각날 정도로 인상적임. 오래된 파리역사를 잘 살러서 멋진 분위기를 연출했으며, 주변의 다른 호텔 식당보다 맛이 훌륭함. 밸런스 프로그램탓인지 우리 부부 모두 아침 한 끼만 먹어도 하루종일 배가 고프지 않음.'

# 입력 데이터 형태소 분석하기
inputdata = []
morphed_input = rhinoMorph.onlyMorph_list(rn, new_input, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'])
morphed_input = ' '.join(morphed_input)                     # 한 개의 문자열로 만들기
inputdata.append(morphed_input)                               # 분석 결과를 리스트로 만들기
X_input = vect.transform(inputdata)
print(float(grid.predict(X_input)))
result = grid.predict(X_input) # 0은 부정,1은 긍정
if result == 0:
    print("부정적인 글입니다")
else:
    print("긍정적인 글입니다")

new_input = '뷰방 예약했는데 이상한 다락방 줘서 첫날부터 기분 잡치고 다음날부터 좋은방 준다고해서 기대했더니 뷰방보다 좋지도 않음...'

# 입력 데이터 형태소 분석하기
inputdata = []
morphed_input = rhinoMorph.onlyMorph_list(rn, new_input, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'])
morphed_input = ' '.join(morphed_input)                     # 한 개의 문자열로 만들기
inputdata.append(morphed_input)                               # 분석 결과를 리스트로 만들기
X_input = vect.transform(inputdata)
print(float(grid.predict(X_input)))
result = grid.predict(X_input) # 0은 부정,1은 긍정
if result == 0:
    print("부정적인 글입니다")
else:
    print("긍정적인 글입니다")