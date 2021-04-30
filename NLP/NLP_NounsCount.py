from konlpy.tag import Okt
from collections import Counter


filename = r'C:\Users\HANA\PycharmProjects\HANATOUR\NLP\TEXT_Example\etc\event_210125.txt'
f = open(filename, 'r', encoding='utf-8')
text_list = f.read()

okt = Okt()
noun = okt.nouns(text_list)
for i,v in enumerate(noun):
    if len(v)<2:
        noun.pop(i)

count = Counter(noun)

noun_list = count.most_common(1000)
for v in noun_list:
    print(v)