import bs4
import re
from urllib import parse
import requests

#input
date_start = "20210410"
date_end = "20210421"
search = "%BA%CE%B5%BF%BB%EA" #부동산
# search = '%C1%F5%B1%C7' #증권

y1 = date_start[0:4]
m1 = date_start[4:6]
d1 = date_start[6:8]

y2 = date_end[0:4]
m2 = date_end[4:6]
d2 = date_end[6:8]

search_in = search

url = 'https://find.mk.co.kr/new/search.php?pageNum=1&cat=&cat1=&media_eco=&pageSize=10&sub=all&dispFlag=OFF&page=news&s_kwd='+search_in+'&s_page=news&go_page=&ord=1&ord1=1&ord2=0&s_keyword='+search_in+'&period=p_direct&s_i_keyword='+search_in+'&s_author=&y1='+y1+'&m1='+m1+'&d1='+d1+'&y2='+y2+'&m2='+m2+'&d2='+d2+'&ord=1&area=ttbd'
print(url)

req = requests.get(url)
bsObj = bs4.BeautifulSoup(req.content.decode('euc-kr', 'replace'), "html.parser")
tag_total_content = bsObj.find('span', {"class":"class_tit"}).get_text()
total_content = int(re.sub('뉴스 검색결과 \[총 (.*)건.*', r'\1', tag_total_content).replace(',',''))
total_page = int(total_content/20)
print(total_content, total_page)
# selenium_req.close()

LIST_title = []
LIST_time = []
LIST_time_year = []
LIST_time_month = []
LIST_time_day = []
LIST_content = []

# for i in range(1,total_page):
for i in range(1,2):
    url = 'https://find.mk.co.kr/new/search.php?pageNum='+str(i)+'&cat=&cat1=&media_eco=&pageSize=20&sub=news&dispFlag=OFF&page=news&s_kwd=' + search_in + '&s_page=total&go_page=page&ord=1&ord1=1&ord2=0&s_keyword=' + search_in + '&y1=' + y1 + '&m1=' + m1 + '&d1=' + d1 + '&y2=' + y2 + '&m2=' + m2 + '&d2=' + d2 + '&area=ttbd'
    print(url)

    req = requests.get(url)
    bsObj = bs4.BeautifulSoup(req.content.decode('euc-kr', 'replace'), "html.parser")
    sub_title = bsObj.find_all('div', {"class":"sub_list"})

    for sub_title_content in sub_title:
        p_title = sub_title_content.find('span', {"class": "art_tit"}).get_text()
        p_time = sub_title_content.find('span', {"class": "art_time"}).get_text()
        p_content = sub_title_content.find('a').get_text()
        p_detail_url = sub_title_content.find('span', {"class": "art_tit"}).a.attrs['href']
        selenium_detail_req = requests.get(p_detail_url)
        bsObj_detail = bs4.BeautifulSoup(selenium_detail_req.content.decode('euc-kr', 'replace'), "html.parser")

        p_detail_title = bsObj_detail.find('h1', {"class": "top_title"}).get_text()

        p_detail_time = ""
        if bsObj_detail.find('li', {"class": "lasttime1"}) is not None:
            p_detail_time = bsObj_detail.find('li', {"class": "lasttime1"}).get_text().replace('입력 : ', '')
        elif bsObj_detail.find('li', {"class": "lasttime"}) is not None:
            p_detail_time = bsObj_detail.find('li', {"class": "lasttime"}).get_text().replace('입력 : ', '')

        p_detail_content = ""
        if bsObj_detail.find('div', {"class": "art_txt"}).td is not None:
            p_detail_content = bsObj_detail.find('div', {"class": "art_txt"}).td.get_text().strip().replace('\n', '')
        else:
            p_detail_content = bsObj_detail.find('div', {"class": "art_txt"}).get_text().strip().replace('\n', '')

        if 'https://www.mk.co.kr/news/stock/view' in p_detail_url:
            selenium_detail_req.close()
            continue

        LIST_time_year.append(p_detail_time.split(' ')[0].split('.')[0])
        LIST_time_month.append(p_detail_time.split(' ')[0].split('.')[1])
        LIST_time_day.append(p_detail_time.split(' ')[0].split('.')[2])

        LIST_title.append(p_detail_title)
        LIST_time.append(p_detail_time)
        LIST_content.append(p_detail_content)

        print(p_detail_url)x
        print(p_detail_title)
        print(p_detail_time)
        print(p_detail_content)

        selenium_detail_req.close()
        # bsObj_detail.close()

import pandas as pd
df = pd.DataFrame(list(zip(LIST_title, LIST_content, LIST_time_year, LIST_time_month,  LIST_time_day)), columns = ['Title', 'Content', 'Year', 'Month', 'Day'])
print(df.head())

df.to_csv("C:\\Users\\HANA\\Downloads\\df.csv",index=True, encoding='utf-8-sig')