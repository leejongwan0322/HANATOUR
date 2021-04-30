from konlpy.tag import *

import csv
import NLP.NPL_library as NPL_library

hannanum = Hannanum()
hannanum.pos  # 형태소 분석과 태깅

word =[]
kkeyword_input = []

#C:\Users\HANA\Desktop\log
#화일을 읽는다.

file_read_csv = 'D:\\test.csv'
flie_save_csv = 'D:\\test_result.csv'

with open(file_read_csv, newline='') as csvfile:
    reader_ilne = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in reader_ilne:
        # print(row)
        # print(row, row[0].replace('"','').replace("_", ' '))
        # kkeyword_input.append(row[0].replace('"','').replace("_", ' ').replace("+", ' '))
        kkeyword_input.append(row[0])

#NLP
f = open(flie_save_csv, 'w')
for q in kkeyword_input:
    # print("input: ", q)
    original_keyword = q.replace('\"','')
    keyword_analysis = hannanum.pos(q.replace('"','').replace("_", ' ').replace("+", ' ').strip())

    for keyword in keyword_analysis:

        # print(keyword, ": ", keyword[0], keyword[1])
        data = ""

        # print(original_keyword)
        keyin_word = original_keyword
        npl_word = NPL_library.check_keyword(original_keyword, keyword)
        npl_kind = keyword[1]

        if npl_word != "" and len(original_keyword) <= 50:
            data = ('insert into DH_NPL_WORD_ARRANGE values (\''+original_keyword + '\',\''+npl_word+'\',\''+npl_kind+'\');\n')
            # data = (original_keyword + '    ' + npl_word + '\n')
            print(data)
            f.write(data)

f.close()