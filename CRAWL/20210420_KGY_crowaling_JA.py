from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote
import pandas as pd
import re

TARGET_URL = 'https://news.joins.com/Search/JoongangNews?page=1&Keyword=%EC%9C%A0%ED%86%B5%EC%97%85&PeriodType=DirectInput&StartSearchDate=03%2F31%2F2015%2000%3A00%3A00&EndSearchDate=03%2F31%2F2021%2000%3A00%3A00&SortType=New&SearchCategoryType=JoongangNews'
TITLE_OF_ARTICLE = []
CONTENT_OF_ARTICLE = []
DATE_OF_ARTICLE = []

def get_link_from_news_title(page_num, URL):
   for i in range(page_num):
    current_page_num = 1 + i
    position = TARGET_URL.index('page=')
    URL_with_page_num = TARGET_URL[:position+5] + str(current_page_num) + TARGET_URL[position+5:] #검색페이지URL
    source_code_form_url = urllib.request.urlopen(URL_with_page_num)
    soup = BeautifulSoup(source_code_form_url, 'lxml', from_encoding='utf-8')
    for title in soup.select('h2.headline.mg'):
      url_link = title.select('a')
      article_URL = url_link[0]['href']
      get_text(article_URL)


def get_text(URL):
  source_code_form_url = urllib.request.urlopen(URL)
  soup = BeautifulSoup(source_code_form_url, 'lxml', from_encoding='utf-8')

  title_link = soup.select('#article_title') #제목 크롤링
  TITLE_OF_ARTICLE.append(title_link[0].string) #내용 쌓기ef get_text(URL):

  date_url = soup.select('div.byline') #날짜 크롤링
  date_of_article = date_url[0].select('em')
  DATE_OF_ARTICLE.append(date_of_article[1].string)

  content = soup.select('div.article_body') #내용 크롤링
  for item in content :
    string_item = str(item.find_all(text=True))
    CONTENT_OF_ARTICLE.append(string_item)

def get_total_num_of_article(URL):
  source_code_from_url = urllib.request.urlopen(URL)
  soup = BeautifulSoup(source_code_from_url, 'lxml', from_encoding='utf-8')
  total_num_of_article = soup.select('span.total_number')
  num = total_num_of_article[0].string
  position = num.index('/') # '/' 위치 지정 #1-81 / 808건
  num = num[position:] #'/' 뒤만 가져오기
  num = re.sub('[^0-9]', '', str(num)) #0-9 사이 숫자 빼고 모두 제거 #808건
  num = int(num) #정수처리 #808
  if(num % 10 >= 1):
    page_num = int((num / 10) + 1)
  else:
    page_num = int(num / 10)
  return page_num #81페이지


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