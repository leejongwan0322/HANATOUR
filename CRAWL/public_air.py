from urllib.parse import urlencode, quote_plus
import requests
import pandas as pd
from bs4 import BeautifulSoup
import bs4

url ="http://apis.data.go.kr/B552584/ArpltnInforInqireSvc"
service_key = "gFDGoRbB6Z0x2falXDyAJ8GotJr14tHzgvYUPbTjOuHFrhFiLpP+DrOGqtrvnP2UxwK+Ij3nS82G7zkL21AV2g=="

base_date = ["202007","202008","202009"]


for i in range(len(base_date)):

    gu_code = '11545' ##구 단위로 데이터를 확보하는 것. ex)11545 = 금천구
    payload = "serviceKey=" + service_key + "&"+"LAWD_CD=" + gu_code + "&"+"DEAL_YMD=" + base_date[i]+ "&"

    res = requests.get(url + payload).text
    xmlobj = bs4.BeautifulSoup(res, 'lxml-xml')
    rows = xmlobj.findAll('item')
    # print(rows)

    rowList = []
    nameList = []
    columnList = []

    rowsLen = len(rows)
    for i in range(0, rowsLen):
        columns = rows[i].find_all()

        columnsLen = len(columns)
        for j in range(0, columnsLen):
            if i == 0:
                nameList.append(columns[j].name)
            eachColumn = columns[j].text
            columnList.append(eachColumn)
        rowList.append(columnList)
        columnList = []

result = pd.DataFrame(rowList, columns=nameList)
print(result.info())


