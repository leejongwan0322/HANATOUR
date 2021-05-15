#https://hansuho113.tistory.com/4?category=913503
from bs4 import BeautifulSoup as bs
from konlpy.tag import Kkma, Okt, Komoran
from konlpy.utils import pprint
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import urllib
import urllib.request as req
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import feedparser

from Common_Module.CMNLP import LemNormalize, get_cluster_details, print_cluster_details

okt = Okt()

TitDesc_list = [
    '트럼프 "대선이후 투표 끔찍" 불복 소송 예고…유혈사태 배제 못한다',
    '개표 뚜껑 열어도 끝날 때까지 끝난 게 아니다'
]

title_list = [
    '도널드 트럼프 미국 대통령이 2020 미 대선 최대 승부처인 펜실베이니아 결과가 나오기도 전에 "때 이른 승리를 선언할 수 있다"는 보도가 나오면서 대선 불복 시나리오가 현실화할 것이란 우려가 커지고 있다.와 미시간,위스콘신,펜실베이니아를 제외한 오하이오아이오와 등 나머지 중서부 ''러스트벨트''에서 승리하거나 우세를 달리는 경우다.',
    '3일(현지시각) 치러질 미국 대통령 선거는 50개주와 수도인 워싱턴을 합친 51곳에서 유권자 전체 투표로 선거인단을 뽑고, 선거인단이 해당 주 유권자들이 선택한 후보를 다시 대통령으로 뽑는 형태다. 메인주와 네브래스카주를 제외한 48개주와 워싱턴은 할당된 선거인단 표를 해당 주에서 이긴 후보에게 모두 몰아주는 ‘승자 독식’ 제도를 택한다. 2016년 대선처럼 선거인단이 많이 배정된 주에서 근소한 차이로 승리한 후보가 전국 투표로 따지면 득표율이 높은 후보에게 승리하는 역전이 일어날 수 있다.'
]

# print(TitDesc_list)
# print(title_list)

TitDesc_okt = []

for item in TitDesc_list:
    # print(item)
    item_nouns = ' '.join(okt.nouns(item))
    # print(item_nouns)
    TitDesc_okt.append(item_nouns)
    # print(TitDesc_okt)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_okt = tfidf_vectorizer.fit_transform(TitDesc_okt)
    # print(tfidf_matrix_okt)

# print(TitDesc_okt[:3])
#Vectorization

# 토큰화 된 문장 리스트를 단어별로 split한 후 2차원 리스트로 저장
WordVoca_list = []
for item in TitDesc_okt:
    # print(item)
    WordVoca_list.append(item.split(' '))

# split된 2차원 리스트 중에서 한 글자짜리 단어들을 모두 제외시키고 WordVoca 리스트 생성
# Word2Vec 모델 학습 데이터로 활용됨
WordVoca = []
for i in range(len(WordVoca_list)):
    element = []
    # print(i)
    for j in range(len(WordVoca_list[i])):
        if len(WordVoca_list[i][j]) > 1:
            element.append(WordVoca_list[i][j])
    # print(element)
    WordVoca.append(element)

print(TitDesc_okt)
print(TitDesc_list)
df_dict = {'TokenizedTitDesc':TitDesc_okt,
           'TitDesc':TitDesc_list}
doc_df = pd.DataFrame(df_dict)
doc_df['title'] = 0
doc_df['num']=0

# print(doc_df)

for i in range(len(doc_df)):
  doc_df.iloc[i, 3] = i
  doc_df.iloc[i, 2] = title_list[i]
doc_df.head()

topic_df = doc_df  # 이후 클러스터별 키워드 추출에 사용될 데이터프레임
print(topic_df)


# x = normalize(tfidf_matrix_okt)
# L2 정규화

# def elbow(normalizedData, Clusters):
#     sse = []
#     for i in range(1,Clusters):
#         kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
#         kmeans.fit(normalizedData)
#         sse.append(kmeans.inertia_)
#     plt.figure(figsize=(7,6))
#     plt.plot(range(1,Clusters), sse, marker='o')
#     plt.xlabel('number of cluster')
#     plt.xticks(np.arange(0,Clusters,1))
#     plt.ylabel('SSE')
#     plt.title('Elbow Method - number of cluster : '+str(Clusters))
#     plt.show()
#
# elbow(tfidf_matrix_okt, 35)

tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize)
feature_vect = tfidf_vect.fit_transform(topic_df['TokenizedTitDesc'])

clusters_num = 2
km_cluster = KMeans(n_clusters=clusters_num, max_iter=10000, random_state=0)
km_cluster.fit(feature_vect)
cluster_label = km_cluster.labels_
cluster_centers = km_cluster.cluster_centers_


topic_df['cluster_label'] = cluster_label
topic_df.head()
for i in range(clusters_num):
  print('<<Clustering Label {0}>>'.format(i)+'\n')
  print(topic_df.loc[topic_df['cluster_label']==i])

cluster_centers = km_cluster.cluster_centers_
print('cluster_centers shape : ', cluster_centers.shape)
print(cluster_centers)

feature_names = tfidf_vect.get_feature_names()
print(topic_df)
print(feature_names)
cluster_details = get_cluster_details(cluster_model=km_cluster, cluster_data=topic_df,\
                                  feature_names=feature_names, clusters_num=clusters_num, top_n_features=10 )
print_cluster_details(cluster_details)


def WordSimilarity(word, count):
    model = Word2Vec(sentences=WordVoca, size=100, window=5, min_count=5, workers=4, sg=1)
    model_result = model.most_similar(positive=[word], topn=count)

    Similarity_df = pd.DataFrame(model_result, columns=[word, 'Similarity'])
    print('{0}과 유사한 단어 Top {1} :'.format(word, count))
    print(Similarity_df)


WordSimilarity('더존', 10)