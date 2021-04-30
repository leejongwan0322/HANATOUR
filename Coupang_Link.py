import requests
import time
import os, sys
import hmac
# import hashlib
import json
import hashlib
import urllib
from urllib import parse

def Coupang_link(access_key, secret_key, link):
    AD_LINK = Get_CoupangAPI_Deeplink(access_key, secret_key, link)
    print(AD_LINK)
    print("결과: 아래 URL을 복사하여 이용하세요.")
    print(get_short_url(AD_LINK["data"][0]['landingUrl']))
    # print(get_short_url('http://sp.moyeola.com/vip.php?method=' + parse.quote(AD_LINK["data"][0]['landingUrl'])))
    # print(get_short_url(link))

def get_short_url(URL):
    client_id = "BvkK2jIhEkDWB3RFzoqn"
    client_secret = "ifH8jl_As1"
    encText = urllib.parse.quote(URL)
    data = "url=" + encText
    url = "https://openapi.naver.com/v1/util/shorturl"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()

    if(rescode==200):
        response_body = response.read()
        data = json.loads(response_body)
        return data['result']['url']
    else:
        print("Error Code:" + rescode)
        return URL

def Get_CoupangAPI_Deeplink(GET_ACCESS_KEY, GET_SECRET_KEY,REQUEST_URL):
    REQUEST_METHOD = "POST"
    DOMAIN = "https://api-gateway.coupang.com"
    URL = "/v2/providers/affiliate_open_api/apis/openapi/v1/deeplink"

    # Replace with your own ACCESS_KEY and SECRET_KEY
    ACCESS_KEY = GET_ACCESS_KEY
    SECRET_KEY = GET_SECRET_KEY

    REQUEST = {"coupangUrls": [REQUEST_URL]}

    def generateHmac(method, url, secretKey, accessKey):
        path, *query = url.split("?")
        os.environ["TZ"] = "GMT+0"
        datetime = time.strftime('%y%m%d') + 'T' + time.strftime('%H%M%S') + 'Z'
        message = datetime + method + path + (query[0] if query else "")

        signature = hmac.new(bytes(secretKey, "utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()
        return "CEA algorithm=HmacSHA256, access-key={}, signed-date={}, signature={}".format(accessKey, datetime, signature)

    authorization = generateHmac(REQUEST_METHOD, URL, SECRET_KEY, ACCESS_KEY)
    url = "{}{}".format(DOMAIN, URL)
    resposne = requests.request(method=REQUEST_METHOD, url=url,headers={
                                    "Authorization": authorization,
                                    "Content-Type": "application/json"
                                },
                                data=json.dumps(REQUEST)
                                )
    return resposne.json()

Coupang_link('f367c021-7514-463c-b8c2-44f1d9031a4b',
             '813ea4854bca9a18365fae579c170cb980e9dda4',
             'https://www.coupang.com/np/search?component=&q=%EC%86%A1%EC%9D%B4%EB%B2%84%EC%84%AF&channel=user')

# http://me2.do/xy2AhDjC #애플
# http://me2.do/5lop26SJ 하이뮨 셀렉스
# https://pages.coupang.com/p/15921
# http://me2.do/FN2qlqCg 휴대폰

