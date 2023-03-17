# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:45:46 2023

@author: 김충환
"""



import pandas as pd
import requests

url = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"

params = {
    "serviceKey": "t6/gaaB02N4JxaUblFxqzO9HBFssqAG/DNrsK+WS9XcziRaUdWgmyiqa9P0wa/1jHYa7RBo9kmtqxU+BQt/EEA==",
    "numOfRows": "5",
    "pageNo": "1",
    "resultType": "json",
    "mrktCls": "KOSPI"
}

response = requests.get(url, params=params)
data = response.json()


# JSON 데이터를 pandas DataFrame으로 변환
df = pd.json_normalize(data["response"]["body"]["items"]["item"])
df.info()

# 결과 출력
print(df)


