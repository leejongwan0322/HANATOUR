from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import Common_Module.CMNLP as CMNLP

#모토사이클, 야구, 그래픽스, 윈도우즈, 중동, 기독교, 전자공학, 의학 8개 주제를 추출
cats = ['rec.motorcycles', 'rec.sport.baseball', 'comp.graphics', 'comp.windows.x', 'talk.politics.mideast', 'soc.religion.christian', 'sci.electronics', 'sci.med']

#위에서 cats변수로 기재된 카테고리만 추출, fetch_20newsgrouops()의 categories에 cats입력
news_df = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), categories=cats, random_state=0)

#LDA는 Count기반의 벡터화만 적용합니다.
count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words='english', ngram_range=(1,2))
fect_vect = count_vect.fit_transform(news_df.data)
print(fect_vect)
print(fect_vect.shape)

print('lda\n')
lda = LatentDirichletAllocation(n_components=8, random_state=0)
lda.fit(fect_vect)
print(lda.components_)
print(lda.components_.shape)

#CountVectorizer객체 내의 전체 word의 명칭을 get_feature_names()을 통해 추출
feature_names = count_vect.get_feature_names()

#토픽별 가장 연관도가 높은 word를 15개만 추출
CMNLP.display_topics(lda, feature_names, 15)