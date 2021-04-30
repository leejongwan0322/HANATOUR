import bs4
import re
from urllib import parse
import requests

LIST_content = []
LIST_title = []
LIST_topic = []

for i in range(1, 2):
    print("page --:", i)
    url = 'https://kin.naver.com/search/list.nhn?query=%EB%8C%80%ED%95%99%EC%83%9D%EC%A7%84%EB%A1%9C%EC%83%81%EB%8B%B4&page=' + str(i)

    req = requests.get(url)
    bsObj = bs4.BeautifulSoup(req.content, "html.parser")
    nk_list = bsObj.find('ul', {"class":"basic1"}).find_all('li')

    for nk_detail in nk_list:
        nk_detail_url = nk_detail.find('a', {"class":"_nclicks:kin.txt _searchListTitleAnchor"})['href']

        req_detail = requests.get(nk_detail_url)
        bsObj_detail = bs4.BeautifulSoup(req_detail.content, "html.parser")
        nk_title = bsObj_detail.find('div', {"class": "title"}).get_text().strip()

        nk_content = ""
        if bsObj_detail.find('div', {"class": "c-heading__content"}) is not None:
            nk_content = bsObj_detail.find('div', {"class": "c-heading__content"}).get_text().strip()
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

df.to_csv("C:\\Users\\HANA\\Downloads\\df.csv",index=True, encoding='utf-8-sig')