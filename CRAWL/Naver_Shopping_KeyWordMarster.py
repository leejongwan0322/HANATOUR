#https://testmanager.tistory.com/116

import bs4
import pandas as pd
from selenium import webdriver
import time
import re
from selenium.webdriver.remote.webelement import WebElement, By
from selenium.webdriver.support.ui import Select

# options = webdriver.ChromeOptions()
# options.add_argument('headless')
# driver = webdriver.Chrome('D:\Chromdriver\chromedriver.exe', options=options)
# driver.get('https://datalab.naver.com/shoppingInsight/sCategory.naver')


# 분야 선택
# driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[1]/span').click()
# driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[1]/ul/li[5]/a').click()
#
# driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[2]/span').click()
# driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[2]/ul/li[3]/a').click()


# 기기, 성별, 연령 전체 체크
# button1 = driver.find_element_by_xpath('//*[@id="18_device_0"]')
# button2 = driver.find_element_by_xpath('//*[@id="19_gender_0"]')
# button3 = driver.find_element_by_xpath('//*[@id="20_age_0"]')
# button1.click()
# button2.click()
# button3.click()

# 조회하기
# button4 = driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/a')
# button4.click()

# keyword = []
# for i in range(1, 2, 1):
#     bsObj = bs4.BeautifulSoup(driver.page_source, "html.parser")
#     top_100_list = bsObj.find('ul', {'class':'rank_top1000_list'}).find_all('li')
#
#     for content in top_100_list:
#         input_test = content.get_text().strip()
#
#         i = 0
#         return_text = ""
#         for input_test_one in input_test:
#             if i <= 2 and input_test_one.isdigit():
#                 continue
#             else:
#                 return_text = return_text + input_test_one;
#             i+=1
#
#         keyword.append(return_text)
        # print(return_text)
#
#     driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[2]/div[2]/div/div/div[2]/div/a[2]').click()
#     time.sleep(2)
#
# driver.close()
# print(keyword)

# options_kw = webdriver.ChromeOptions()
# options.add_argument('headless')
# driver_kw = webdriver.Chrome('D:\Chromdriver\chromedriver.exe', options=options_kw)
# driver_kw.get('https://searchad.naver.com/login?returnUrl=https:%2F%2Fmanage.searchad.naver.com&returnMethod=get')

# 로그인하기
# driver_kw.find_element_by_xpath('//*[@id="uid"]').click()
# driver_kw.find_element_by_xpath('//*[@id="uid"]').send_keys('leejwwin')
#
# driver_kw.find_element_by_xpath('//*[@id="upw"]').click()
# driver_kw.find_element_by_xpath('//*[@id="upw"]').send_keys('89621638ljw#')
#
# driver_kw.find_element_by_xpath('//*[@id="container"]/div/div/fieldset/div/span/button').click()
# time.sleep(5)

#도구로 이동
# driver_kw.find_element_by_xpath('//*[@id="root"]/div/div[1]/div/div[1]/div[2]/div/div/div[1]/ul/li[4]/div/a').click()
# driver_kw.find_element_by_xpath('//*[@id="root"]/div/div[1]/div/div[1]/div[2]/div/div/div[1]/ul/li[4]/div/div/div[5]/a/button').click()
# time.sleep(3)

#값입력
# word_list = ""
# for i, word in enumerate(keyword):
#     word_list = word_list + "\n" + word
#     if (i+1)%5 == 0:
#         print(word_list)
#         driver_kw.find_element_by_xpath('/html/body/elena-root/elena-wrap/div/div[2]/elena-tool-wrap/div/div/div/div/elena-keyword-planner/div[2]/div[1]/div[1]/div[2]/div/div/div[1]/div/textarea').send_keys(word_list)
#         driver_kw.find_element_by_xpath('/html/body/elena-root/elena-wrap/div/div[2]/elena-tool-wrap/div/div/div/div/elena-keyword-planner/div[2]/div[1]/div[1]/div[3]/button').click()
#         time.sleep(3)
#         driver_kw.find_element_by_xpath('/html/body/elena-root/elena-wrap/div/div[2]/elena-tool-wrap/div/div/div/div/elena-keyword-planner/div[2]/div[1]/div[2]/div[1]/div/button').click()
#         driver_kw.find_element_by_xpath('/html/body/elena-root/elena-wrap/div/div[2]/elena-tool-wrap/div/div/div/div/elena-keyword-planner/div[2]/div[1]/div[1]/div[2]/div/div/div[1]/div/textarea').clear()
#         word_list = ""
#         time.sleep(3)
#
# time.sleep(10)
# driver_kw.close()

import os
import cv2
from openpyxl import load_workbook

all_excel_list = []
path_dir = 'C:\\Users\\HANA\\Downloads'
for root, dirs, files in os.walk(path_dir):
    for fname in files:
        if '연관키워드' in fname:
            full_fname = os.path.join(root, fname)
            print(full_fname)

            load_wb = load_workbook(full_fname, data_only=True)
            load_ws = load_wb['sheet']
            all_values = []

            i = 0;
            for row in load_ws.rows:

                if i==0:
                    i=i+1;
                    continue;

                row_value = []
                for cell in row:
                    row_value.append(cell.value)
                all_values.append(row_value)

            all_excel_list = all_excel_list + all_values
            print(all_values)
        else:
            continue

        # print(full_fname)

# print(all_excel_list, len(all_excel_list))
# print(all_excel_list[0][0])

i=0
while i <= len(all_excel_list)-1:
    print(all_excel_list[i][0])
    i=i+1

    #이곳에 셀러마스터 기능을 넣을것