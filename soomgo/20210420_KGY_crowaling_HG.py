import bs4
import re
from urllib import parse
import requests
import pandas as pd
import time

date_start = "2020.03.31"
date_end = "2021.03.31"
keyword = "%EC%9C%A0%ED%86%B5%EC%97%85"

url = 'https://search.hankyung.com/apps.frm/search.news?query='+keyword+'&sort=DATE%2FDESC%2CRANK%2FDESC&period=DATE&area=ALL&mediaid_clust=HKPAPER%2CHKCOM&sdate='+date_start+'&edate='+date_end+'&exact=&include=&except=&page='

print(url)

r = requests.get(url)
soup = bs4.BeautifulSoup(r.content.decode('euc-kr', 'replace'), "html.parser")
tag_total_content = soup.find('h3', {"class":"tit"}).select('span')[1].get_text()
total_content = int(re.sub('.*[/](.*)..',r'\1', tag_total_content).replace(',',''))
total_page = int(total_content/10)

print(total_content, total_page)

LIST_title = []
LIST_time = []
LIST_time_year = []
LIST_time_month = []
LIST_time_day = []
LIST_content = []

for i in range(1, total_page):
    url = 'https://search.hankyung.com/apps.frm/search.news?query=' + keyword + '&sort=DATE%2FDESC%2CRANK%2FDESC&period=DATE&area=ALL&mediaid_clust=HKPAPER%2CHKCOM&sdate=' + date_start + '&edate=' + date_end + '&exact=&include=&except=&page=' + str(i)
    # print(url)

    r = requests.get(url)
    soup = bs4.BeautifulSoup(r.content.decode('euc-kr', 'replace'), "html.parser")
    sub_title = soup.find_all('div', {"class": "txt_wrap"})

    for sub_title_content in sub_title:
        p_detail_url = sub_title_content.a.attrs['href']
        req2 = requests.get(p_detail_url)
        time.sleep(10)
        soup_detail = bs4.BeautifulSoup(req2.content.decode('euc-kr', 'replace'), "html.parser")
        print('--------')
        print(p_detail_url)
        print(soup_detail)

        p_detail_title = soup_detail.find('h1', {"class": "title"})
        # print(p_detail_title)
        # p_detail_time = soup_detail.find('span',{"class": "date-published"}), get_text().replace('입력 : ', '')
        # p_detail_content = soup_detail.find('div',{"class":"articletxt"}), get_text()

        # List_time_year.append(p_detail_time.split(''[0]).split(',')[0]
        # List_time_month.append(p_detail_time.split(''[0]).split(',')[1]
        # List_time_day.append(p_detail_time.split(''[0]).split(',')[2]

        # List_title.append(p_detail_title)
        # List_time.append(p_detail_time)
        # List_content.append(p_detail_content)

        # print(p_detail_url)
        # print(p_detail_title)
        # print(p_detail_time)
        # print(p_detail_content)
        # req2.close()

        # exit()
