import bs4
from selenium import webdriver
import re

def Get_html_from_Selenium(url):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')

    #https://chromedriver.chromium.org/downloads
    driver = webdriver.Chrome('D:\Chromdriver\chromedriver.exe', options=options)

    driver.get(url)
    return driver

url = 'https://www.albamon.com/list/gi/mon_area_list.asp?page=1&ps=20&ob=6&lvtype=1&rArea=,I000,&rWDate=1&Empmnt_Type='
selenium_req = Get_html_from_Selenium(url)

bsObj = bs4.BeautifulSoup(selenium_req.page_source, "html.parser")
albamon_list = bsObj.find_all('button', {'onclick': re.compile('getTogether_gi.*')})

p_area = ""
for albamon_info in albamon_list:
    code = 'dev_preview__' + albamon_info['onclick'].replace('getTogether_gi', '').split("'")[1]
    code_html = bsObj.find('tr', {'id': code})
    print(code_html.find('td', {'class': 'area'}).get_text().replace('스크랩','').strip())
    print(code_html.find('p', {'class': 'cName'}).get_text())
    print(code_html.find('p', {'class': 'cTit'}).get_text())
    print(code_html.find('td', {'class': 'pay'}).get_text())
    print('--------------')

selenium_req.close()
bsObj.clear()