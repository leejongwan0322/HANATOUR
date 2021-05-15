from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote
import pandas as pd
import re

TARGET_URL = 'https://news.joins.com/Search/JoongangNews?page=1&Keyword=%EC%9C%A0%ED%86%B5%EC%97%85&PeriodType=DirectInput&StartSearchDate=03%2F31%2F2015%2000%3A00%3A00&EndSearchDate=03%2F31%2F2021%2000%3A00%3A00&SortType=New&SearchCategoryType=JoongangNews'
TITLE_OF_ARTICLE = []
CONTENT_OF_ARTICLE = []
DATE_OF_ARTICLE = []

page_num = get_total_num_of_article(TARGET_URL)
print(page_num)

get_link_from_news_title(50, TARGET_URL) #2페이지 실행
df = pd.DataFrame(list(zip(TITLE_OF_ARTICLE, CONTENT_OF_ARTICLE, DATE_OF_ARTICLE)), columns = ['Title', 'Content','Date'])
df.head() #상위권 출력


df['Date'] = df['Date'].str.replace('[^0-9]', '', regex=True) #숫자만 가져오기
df['Date'] = pd.to_datetime(df['Date'],format='%Y%m%d%H%M') # 연, 월, 일, 시간으로 정리
df['Year'] = df['Date'].dt.year #연도
df['Month'] = df['Date'].dt.month #월
df['Day'] = df['Date'].dt.day #일
df = df.drop(['Date'],axis=1) #날짜 열로 나타내기
df['Content'] = df['Content'].str.replace(r"([a-zA-Z0-9_.+1]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", '')

df['Content'] = df['Content'].str.replace('[', '')
df['Content'] = df['Content'].str.replace(']', '')
df['Content'] = df['Content'].str.replace("'", "")
df['Content'] = df['Content'].str.replace('"', '')
df['Content'] = df['Content'].str.replace(',', '')
df['Content'] = df['Content'].str.replace(r'\\n', '')
df['Content'] = df['Content'].str.replace(r'\\r', '')
df['Content'] = df['Content'].str.replace(r'\\xa0', '')
df['Content'] = df['Content'].str.replace('‘', '')
df['Content'] = df['Content'].str.replace('’', '')
df['Content'] = df['Content'].str.replace('“', '')
df['Content'] = df['Content'].str.replace('”', '')
df['Content'] = df['Content'].str.replace('.', '')
df['Content'] = df['Content'].str.replace('//', '')
df['Content'] = df['Content'].str.replace('ㆍ', '')
df['Content'] = df['Content'].str.replace('  ', '')
df['Content'] = df['Content'].str.replace('·', '')
df['Content'] = df['Content'].str.replace('…', '')
df['Content'] = df['Content'].str.replace('〈br〉', '')
df['Content'] = df['Content'].str.replace('「', '')
df['Content'] = df['Content'].str.replace('」', '')
df['Content'] = df['Content'].str.replace('<>', '')
df['Content'] = df['Content'].str.replace(' 〈〉', '')
df['Content'] = df['Content'].str.replace('■', '')
df['Content'] = df['Content'].str.replace('※', '')
df['Content'] = df['Content'].str.replace(r'\\', '')
df['Content'] = df['Content'].str.replace('아티클 공통 : DA 250', '')
df['Content'] = df['Content'].str.replace('아티클 공통 : 관련기사', '')
df['Content'] = df['Content'].str.replace('중앙일보디자인=', '')
df['Content'] = df['Content'].str.replace('그래픽=*', '')
df['Content'] = df['Content'].str.replace('기자', '')
df['Content'] = df['Content'].str.lstrip()

df['Content'] = df['Content'].str.replace(r'\(([^)]+)\)','')
df['Content'] = df['Content'].str.replace(r'〈([^〉]+)〉','')

df['Title'] = df['Title'].str.replace(r'\[([^)]+)\]','') #[단독]제거
df['Title'] = df['Title'].str.replace('‘', '')
df['Title'] = df['Title'].str.replace('’', '')
df['Title'] = df['Title'].str.replace(',', '')
df['Title'] = df['Title'].str.replace('[', '')
df['Title'] = df['Title'].str.replace(']', '')
df['Title'] = df['Title'].str.replace('…', '')
df['Title'] = df['Title'].str.replace('-', '')
df['Title'] = df['Title'].str.replace('·', '')
df['Title'] = df['Title'].str.replace('“', '')
df['Title'] = df['Title'].str.replace('”', '')
df['Title'] = df['Title'].str.replace('"', '')
df['Title'] = df['Title'].str.replace('···', '')
df['Title'] = df['Title'].str.replace("'", '')
df['Title'] = df['Title'].str.replace('?', '')
df['Title'] = df['Title'].str.lstrip()

df.head() #상위권 출력

df.to_csv("df.csv",index=True, encoding='utf-8-sig')