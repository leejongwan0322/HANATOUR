import bs4
import re
from urllib import parse
import requests

LIST_content = []
LIST_title = []
LIST_topic = []

for i in range(1, 100):
    print("page --:", i)
    url = 'https://search.naver.com/search.naver?where=view&sm=tab_jum&query=%EC%95%A1%ED%8A%B8%EC%98%A4%EB%B8%8C%ED%82%AC%EB%A7%81'

    req = requests.get(url)
    bsObj = bs4.BeautifulSoup(req.content, "html.parser")
    nk_list = bsObj.find('ul', {"class":"lst_total _list_base"})

    for nk_detail in nk_list:
        nk_detail_url = nk_detail.find('a', {"class":"api_txt_lines total_tit"})['href']

        req_detail = requests.get(nk_detail_url)
        bsObj_detail = bs4.BeautifulSoup(req_detail.content, "html.parser")
        nk_title = bsObj_detail.find('div', {"class": "title"}).get_text().strip()
        # print("title: ", nk_title)

        nk_content = ""
        if bsObj_detail.find('div', {"class": "lst_total _list_base"}) is not None:
            nk_content = bsObj_detail.find('div', {"class": "total_group"}).get_text().strip()
        else:
            nk_content = ""

        # print("content: ",nk_content)

        nk_topic = bsObj_detail.find('a', {"class": "tag-list__item tag-list__item--category"}).get_text().strip().replace('태그 디렉터리Ξ ', '')
        # print(nk_topic)

        LIST_title.append(nk_title)
        LIST_content.append(nk_content)
        LIST_topic.append(nk_topic)


import pandas as pd
df = pd.DataFrame(list(zip(LIST_title, LIST_content, LIST_topic)), columns = ['Title', 'Content', 'Topic'])
print(df.head())

df.to_csv("df.csv",index=True, encoding='utf-8-sig')