from urllib.parse import urlencode, quote_plus
import requests
import pandas as pd
from bs4 import BeautifulSoup
import bs4

url ="http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
service_key = "gFDGoRbB6Z0x2falXDyAJ8GotJr14tHzgvYUPbTjOuHFrhFiLpP%2BDrOGqtrvnP2UxwK%2BIj3nS82G7zkL21AV2g%3D%3D"

base_date = ["종로구","중구"]

for i in range(len(base_date)):

    gu_code = '11545' ##구 단위로 데이터를 확보하는 것. ex)11545 = 금천구
    payload = "?serviceKey=" + service_key + "&"+"returnType=xml&numOfRows=100000&pageNo=1&dataTerm=3MONTH&&ver=1.0&stationName=" + base_date[i]
    res = requests.get(url + payload).text
    xmlobj = bs4.BeautifulSoup(res, 'lxml-xml')
    rows = xmlobj.findAll('item')
    print(rows)

    # exit()
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


