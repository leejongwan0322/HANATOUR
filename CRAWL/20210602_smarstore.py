import bs4
import pandas as pd
from selenium import webdriver
import time
import re
import json
from selenium.webdriver.remote.webelement import WebElement, By
from selenium.webdriver.support.ui import Select
from urllib.parse import urlparse, parse_qs

def GetNetworkResources(driver):
    Resources = driver.execute_script("return window.performance.getEntries();")
    originProductNo = ''
    merchantNo = ''

    for resource in Resources:
        # print(resource)
        # print(resource['name'])
        # print(urlparse(resource['name']))
        # print(parse_qs(urlparse(resource['name']).query))

        if 'originProductNo' in parse_qs(urlparse(resource['name']).query).keys():
            originProductNo =  (parse_qs(urlparse(resource['name']).query)['originProductNo'][0])
        if 'merchantNo' in parse_qs(urlparse(resource['name']).query).keys():
            merchantNo =  (parse_qs(urlparse(resource['name']).query)['merchantNo'][0])

    return [originProductNo, merchantNo]

url_list = [
    'https://smartstore.naver.com/beautycom/products/3694521897',
    'https://smartstore.naver.com/beautycom/products/5009598668',
    'https://smartstore.naver.com/beautycom/products/3694591982',
    'https://smartstore.naver.com/beautycom/products/4017256136',
    'https://smartstore.naver.com/beautycom/products/3695183206',
    'https://smartstore.naver.com/botem/products/2895053496',
    'https://smartstore.naver.com/botem/products/2895058949',
    'https://smartstore.naver.com/botem/products/2132596698',
    'https://smartstore.naver.com/botem/products/2132610814',
    'https://smartstore.naver.com/dreamsys_kr/products/4675268827',
    'https://smartstore.naver.com/dreamsys_kr/products/4916403013',
    'https://smartstore.naver.com/dreamsys_kr/products/4841056986',
    'https://smartstore.naver.com/dreamsys_kr/products/4600838467',
    'https://smartstore.naver.com/beautyicon/products/4865940533',
    'https://smartstore.naver.com/beautyicon/products/5118524935',
    'https://smartstore.naver.com/beautyicon/products/5110494145',
    'https://smartstore.naver.com/beautyicon/products/5216102439',
    'https://smartstore.naver.com/beautyicon/products/4864956164',
    'https://smartstore.naver.com/beautyicon/products/5127874163',
    'https://smartstore.naver.com/beautyicon/products/4609252831',
    'https://smartstore.naver.com/beautyicon/products/4866127777',
    'https://smartstore.naver.com/beautyicon/products/4976132422',
    'https://smartstore.naver.com/beautyicon/products/5052407363',
    'https://smartstore.naver.com/beautyicon/products/4865026141',
    'https://smartstore.naver.com/beautyicon/products/4864871119',
    'https://smartstore.naver.com/beautyicon/products/4662369261',
    'https://smartstore.naver.com/beautyicon/products/5127882153',
    'https://smartstore.naver.com/beautyicon/products/5216103275',
    'https://smartstore.naver.com/vividoanon/products/2496749609',
    'https://smartstore.naver.com/vividoanon/products/4410400314',
    'https://smartstore.naver.com/vividoanon/products/2993147179'
]

rating = []
review = []

for i in range(0,len(url_list),1):
    print(url_list[i])
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome('D:\Chromdriver\chromedriver.exe', options=options)
    driver.get(url_list[i])
    bsObj = bs4.BeautifulSoup(driver.page_source, "html.parser")

    # print(bsObj)
    # accountNo = re.search(r'"accountNo":(.........+?)', bsObj.text, re.S)
    # json_string = accountNo.group(1)
    # accountNo = json.loads(json_string)
    # print(accountNo)

    # productID = re.search(r'"productNo":"(..........+?)', bsObj.text, re.S)
    # json_string = productID.group(1)
    # productID = json.loads(json_string)
    # print(productID)

    page_info = GetNetworkResources(driver)
    # print(page_info)
    productID = page_info[0]
    accountNo = page_info[1]

    for i in range(1,500,1):
        url = 'https://smartstore.naver.com/i/v1/reviews/paged-reviews?page='+str(i)+'&merchantNo='+str(accountNo)+'&originProductNo='+str(productID)+'&sortType=REVIEW_RANKING'
        print(url)
        driver_url = webdriver.Chrome('D:\Chromdriver\chromedriver.exe', options=options)
        driver_url.get(url)
        bsObj_url = bs4.BeautifulSoup(driver_url.page_source, "html.parser")
        driver_url.close()

        if bsObj_url.text == 'OK':
            print("OK: ", url)
            break

        res = json.loads(bsObj_url.text)
        for i in range(0,len(res['contents'])):
            print(res['contents'][i]['reviewScore'], res['contents'][i]['reviewContent'].replace('\n', ' '))
            rating.append(res['contents'][i]['reviewScore'])
            review.append(res['contents'][i]['reviewContent'].replace('\n', ' '))

    driver.close()

column_list = ["rating","review"]
df = pd.DataFrame(list(zip(rating, review)), columns=column_list)
df.to_csv("review_smartstore.csv",index=True, encoding='utf-8-sig')

data_smart = pd.read_csv('./review_smartstore.csv')
# print(data_smart.info())
data_smart_positive = data_smart[data_smart['rating'] >= 4]
data_smart_negative = data_smart[data_smart['rating'] < 4]

positive_review_list = data_smart_positive['review'].values.tolist()
negative_review_list = data_smart_negative['review'].values.tolist()

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font="Malgun Gothic", rc={"axes.unicode_minus":False},style='darkgrid')
okt = Okt()

positive_noun = okt.nouns(' '.join(positive_review_list))
positive_noun = Counter(positive_noun)
noun_positive_list = positive_noun.most_common(20)

print(type(noun_positive_list))
print(noun_positive_list)
# print([item for t in noun_positive_list for item in t])
print([t[0] for t in noun_positive_list])
print([t[1] for t in noun_positive_list])

plt.bar(list(range(len(noun_positive_list))), [t[1] for t in noun_positive_list])
plt.title('Positive', fontsize=10)
plt.xlabel('# of count', fontsize=10)
plt.ylabel('word', fontsize=10)
plt.xticks(list(range(len(noun_positive_list))), [t[0] for t in noun_positive_list], fontsize=7)
plt.show()

# negative_noun = okt.nouns(' '.join(negative_review_list))
# negative_noun = Counter(negative_noun)
# noun_negative_list = negative_noun.most_common(100)
# print(noun_negative_list)

# plt.bar(list(range(len(noun_negative_list))), [t[1] for t in noun_negative_list])
# plt.title('Negative', fontsize=10)
# plt.xlabel('# of count', fontsize=10)
# plt.ylabel('word', fontsize=10)
# plt.xticks(list(range(len(noun_negative_list))), [t[0] for t in noun_negative_list], fontsize=7)
# plt.show()