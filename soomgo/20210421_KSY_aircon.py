import csv
import json

# 문제1. aircon파일을 불러와 형태소 분석 후 json 형태 파일로 저장하세요.
# 핵심포인트 encoding에 UTF8과 dump시 ensure_ascii=false를 하지 않으면 json화일이 깨짐
file_in_csv = open('D:\\#프로젝트\\20210419_KSY_aircon\\aircon.csv', 'r', encoding='UTF8')
file_out_json = open('D:\\#프로젝트\\20210419_KSY_aircon\\aircon.json', 'w+', encoding='UTF8')

with file_in_csv as input_file, file_out_json as output_file:
    reader = csv.reader(input_file)

    col_names = next(reader)
    print(col_names)

    #그 다음 줄부터 zip으로 묶어서 json dumps
    for cols in reader:
        print(col_names)
        print(cols)
        print(zip(col_names, cols))
        exit()
        doc = {col_name: col for col_name, col in zip(col_names, cols)}
        print(json.dumps(doc, ensure_ascii=False, indent="\t"), file=output_file)
        # print(json.dumps(doc, ensure_ascii=False, indent="\t"), file=output_file)

file_in_csv.close()
file_out_json.close()

# 문제2. 1을 통해 생성된 파일에 대해 ["Noun","Verb","Adverb","Adjective"]품사만 남기고, TF-IDF 기반의 term document matrix를 생성하여 저장하세요.(이 때 상위 빈도 50개의 형태소만 저장,정규화 방법 norm1 적용)
file_read_json = open('D:\\#프로젝트\\20210419_KSY_aircon\\aircon.json', 'r', encoding='utf8')
with file_read_json as json_file:
    data = json.loads("[" + json_file.read().replace("}\n{", "},\n{").replace("\ufeff", "") + "]")
file_read_json.close()

sentense_all = ""
for i in range(0, len(data)):
    # print(data[i]['review'])
    sentense_all = sentense_all + " " + data[i]['review']

# ver1: Hannanum
# from konlpy.tag import Hannanum
# hannanum = Hannanum()
# word_list = sentense_all.split()
# print(sentense_all)
# print(word_list)
# print(hannanum.analyze(sentense_all))

# sentense_analyze = hannanum.analyze(sentense_all)
# for sentense_group in sentense_analyze:
#     for word_group in sentense_group:
#         for word_unit in word_group:
            # print(word_unit[0], word_unit[1])

#Ver2 Okt
from konlpy.tag import Okt
okt = Okt()
sentense_analyze = okt.pos(sentense_all)
print(sentense_analyze)
sentense_all = ""
for sentense_group in sentense_analyze:
    if sentense_group[1] in ('Noun', 'Verb', 'Adverb', 'Adjective'):
        print(sentense_group[0], sentense_group[1])
        sentense_all = sentense_all + " " + sentense_group[0]

import collections
word_list = sentense_all.split()
print(word_list)

word_list_final = collections.Counter(word_list)
print(word_list_final)
print(len(word_list_final))

top_word = []
top_word_count = []
sentense_all = ""
for i in range(0,49):
    print(word_list_final.most_common()[i][0])
    sentense_all = sentense_all + " " + word_list_final.most_common()[i][0]

    top_word.append(word_list_final.most_common()[i][0])
    top_word_count.append(int(word_list_final.most_common()[i][1]))

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [sentense_all]
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())

# print(tfidfv.vocabulary_)
# tfidf_vectorizer.fit(text) # 벡터라이저가 단어들을 학습합니다.
# tfidf_vectorizer.vocabulary_ # 벡터라이저가 학습한 단어사전을 출력합니다.
# sorted(tfidf_vectorizer.vocabulary_.items()) # 단어사전을 정렬합니다.


# 문제3. 문제2를 통해 생성된 상위 빈도 50개의 형태소에 대해 트리맵을 이용해 시각화된 결과를 출력하세요.
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc, font_manager
import squarify

matplotlib.font_manager._rebuild()
squarify.plot(sizes=top_word_count, label=top_word_count, alpha=.7 )
plt.axis('off')
plt.show()

