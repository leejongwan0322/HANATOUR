import bs4
import re
import time
from urllib import parse
import requests
from selenium import webdriver

def Get_html_from_Selenium(url):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')

    #https://chromedriver.chromium.org/downloads
    driver = webdriver.Chrome('D:\Chromdriver\chromedriver.exe', options=options)
    driver.get(url)

    return driver

url = 'https://search.naver.com/search.naver?where=view&sm=tab_jum&query=%EA%B0%80%EC%83%81%ED%99%94%ED%8F%90'
target_driver = Get_html_from_Selenium(url)
target_bsObj = bs4.BeautifulSoup(target_driver.page_source, "html.parser")
target_list = target_bsObj.find_all('div', {'class','total_wrap api_ani_send'})

for target in target_list:
    url = target.a.attrs['data-url']
    if 'cafe.naver' in url:
        continue

    print(url)
    blog_driver = Get_html_from_Selenium(url)
    blog_driver.switch_to.frame('mainFrame')

    bsObj = bs4.BeautifulSoup(blog_driver.page_source, "html.parser")

    tit = bsObj.find("meta", {"property": "og:title"}).attrs['content']
    title = tit
    print(tit)

    overlays = ".nick"
    nick = blog_driver.find_element_by_css_selector(overlays)
    nickname = nick.text
    # print(nickname)

    # overlays = '.se-component se-text se-l-default'
    # contents = blog_driver.find_element_by_css_selector(overlays)
    # content_list = []
    # for content in contents:
    #     content_list.append(content.text)
    #
    # content_str = ' '.join(content_list)

    content_list = []
    for content in bsObj.find_all('p', {'class': re.compile('se-text-paragraph')}):
        content_list.append(content.text)

    content_str = ' '.join(content_list).replace(title, '')
    print(content_str)

    # print(title, nickname, content_str)
