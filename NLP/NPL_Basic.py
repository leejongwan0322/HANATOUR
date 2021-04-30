from nltk import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
import nltk

from konlpy.tag import Okt
from konlpy.tag import Kkma

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

text_sample = '미국 국제무역위원회(ITC)가 LG화학과 SK이노베이션의 전기차 배터리 영업비밀 침해 소송의 최종 판결일을 12월10일로 또 연기하면서 양측의 합의 여부가 주목받는다. 당초 이날 LG화학 승소로 최종 판결이 나올 것이 유력했으나, '
# text_sample = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."

sentences = sent_tokenize(text=text_sample)
words = word_tokenize(text_sample)
# print(pos_tag(words))

# print(type(sentences), len(sentences))
# print(sentences)

# print(type(words), len(words))

# print(len(nltk.corpus.stopwords.words('english')))
# print(nltk.corpus.stopwords.words('english')[:20])

# print(len(nltk.corpus.stopwords.words('korea')))
# print(words)
# word_tokens = CMNLP.tokenize_text(str(text_sample))
# print(type(word_tokens), len(word_tokens))

# stopwords = nltk.corpus.stopwords.words('english')
# all_tokens = []
# for sentence in words:
#     filtered_words=[]

print('Okt')
print(Okt().morphs(text_sample))
print(Okt().pos(text_sample))
print(Okt().nouns(text_sample))

print('Kkma')
print(Kkma().morphs(text_sample))
print(Kkma().pos(text_sample))
print(Kkma().nouns(text_sample))

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"
# 위의 불용어는 명사가 아닌 단어 중에서 저자가 임의로 선정한 것으로 실제 의미있는 선정 기준이 아님
stop_words=stop_words.split(' ')
word_tokens = word_tokenize(example)

result = []
for w in word_tokens:
    if w not in stop_words:
        result.append(w)
# 위의 4줄은 아래의 한 줄로 대체 가능
# result=[word for word in word_tokens if not word in stop_words]

print(word_tokens)
print(result)
