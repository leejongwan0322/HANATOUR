#https://testmanager.tistory.com/116

import bs4
import pandas as pd
from selenium import webdriver
import time
import re
from selenium.webdriver.remote.webelement import WebElement, By
from selenium.webdriver.support.ui import Select

options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome('D:\Chromdriver\chromedriver.exe', options=options)
driver.get('https://datalab.naver.com/shoppingInsight/sCategory.naver')


# 분야 선택
driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[1]/span').click()
driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[1]/ul/li[5]/a').click()

driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[2]/span').click()
driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[2]/ul/li[3]/a').click()


# 기기, 성별, 연령 전체 체크
button1 = driver.find_element_by_xpath('//*[@id="18_device_0"]')
button2 = driver.find_element_by_xpath('//*[@id="19_gender_0"]')
button3 = driver.find_element_by_xpath('//*[@id="20_age_0"]')
button1.click()
button2.click()
button3.click()

# 조회하기
button4 = driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/a')
button4.click()

for i in range(1, 27, 1):
    bsObj = bs4.BeautifulSoup(driver.page_source, "html.parser")
    top_100_list = bsObj.find('ul', {'class':'rank_top1000_list'}).find_all('li')

    for content in top_100_list:
        input_test = content.get_text().strip()

        i = 0
        return_text = ""
        for input_test_one in input_test:
            if i <= 2 and input_test_one.isdigit():
                continue
            else:
                return_text = return_text + input_test_one;
            i+=1

        print(return_text)

    driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[2]/div[2]/div/div/div[2]/div/a[2]').click()
    time.sleep(2)

driver.close()