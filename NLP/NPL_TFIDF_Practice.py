from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    '신종 코로나바이러스 감염증(코로나19) 일일 확진자가 8일 0시 기준 143명 발생했다. 전날인 7일 0시 기준 확진자가 나흘만에 두자릿수를 기록했으나, 해외유입을 제외한 지역발생 확진자가 다시 100명을 넘게 나타난 영향이다.',
]
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    '신종 코로나바이러스 감염증(코로나19) 일일 확진자가 8일 0시 기준 143명 발생했다. 전날인 7일 0시 기준 확진자가 나흘만에 두자릿수를 기록했으나, 해외유입을 제외한 지역발생 확진자가 다시 100명을 넘게 나타난 영향이다.',
]
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)