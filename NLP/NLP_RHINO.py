import os

# 파일 읽기 함수 정의
def read_data(filename):
    with open(filename, 'r', encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]

        #파일의 헤더(컬럼명) 제외
        data = data[1:]
    return data

# 파일 쓰기 함수 정의
def write_data(data, filename):
    with open(filename, 'w') as f:
        f.write(data)

data = read_data(r'C:\Users\HANA\PycharmProjects\HANATOUR\NLP\TEXT_Example\ratings.txt')    # 1은 긍정, 0은 부정


import rhinoMorph
rn = rhinoMorph.startRhino()

# 20만 건 전체 문장 형태소 분석. 시간이 많이 소요됨
morphed_data = ''

for data_each in data:
    morphed_data_each = rhinoMorph.onlyMorph_list(rn, data_each[1], pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'])
    joined_data_each = ' '.join(morphed_data_each)
    if joined_data_each: # 내용이 있는 경우만 저장함
        morphed_data += data_each[0]+"\t"+joined_data_each+"\t"+data_each[2]+"\n"   # 원본과 같은 양식으로 만듦

# 형태소 분석된 파일 저장
write_data(morphed_data, r'C:\Users\HANA\PycharmProjects\HANATOUR\NLP\TEXT_Example\ratings_morphed.txt')
