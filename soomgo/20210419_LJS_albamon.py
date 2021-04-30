import bs4
from selenium import webdriver

def Get_html_from_Selenium(url):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')

    #https://chromedriver.chromium.org/downloads
    driver = webdriver.Chrome('D:\Chromdriver\chromedriver.exe', options=options)

    driver.get(url)
    return driver

url = 'https://www.albamon.com/list/gi/mon_area_list.asp?ps=20&ob=6&lvtype=1&rArea=,I000,&rWDate=1&Empmnt_Type='
selenium_req = Get_html_from_Selenium(url)

bsObj = bs4.BeautifulSoup(selenium_req.page_source, "html.parser")
albamon_list = bsObj.find_all('dl', {"class":"toplg"})
# albamon_list = bsObj.find('dl', {"class":"toplg"})

# print(albamon_list)
print("-------------------")

for albamon_info in albamon_list:
    p_company = albamon_info.find('dd',{"class":"cHeader"}).get_text()
    p_title = albamon_info.find('dd',{"class":"cTit iconHurry"}).get_text()

    p_pay = ""
    if albamon_info.find('em') is not None:
        p_pay = albamon_info.find('em').get_text()
    else:
        p_pay =""

    p_address = albamon_info.find('dd',{"class":"cEtc"}).get_text().replace(p_pay + "원", '')
    p_code = albamon_info.find('a',{"class":"pgSimpleGI"}).attrs['id'].replace('pgSimpleImgPL_', '')
    p_url_detail = 'http://www.albamon.com/recruit/view/gi?AL_GI_No='+p_code+'&optgf=ltareatoplogo'

    print("-------------------")
    print(p_company)
    print(p_title)
    print(p_address)
    print(p_pay)
    print(p_code)
    print(p_url_detail)

selenium_req.close()
bsObj.clear()