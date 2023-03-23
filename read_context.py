# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:54:28 2023

@author: kch0325
"""



import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

# csv 파일 읽기
df = pd.read_csv('read_url_fin.csv',index_col=0)
df.info()
# 각 url링크에 접속하고 웹페이지 내용 파싱하여 텍스트 저장
texts = []
for url in df['url']:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    dl_con = soup.find_all('dl','qnaLists')
    for con in dl_con:
        text = con.text.split('질문Q')
        for i in range(1, len(text)):
            qna = text[i].split('보기\n\n\n답변')
            question = qna[0].replace('\n', '')
            answer = qna[1].replace('\n', '').replace('\r', '')
            # 글자수 정보 제거하기
            answer = re.sub(r'글자수\s[\d,]+자[\d,]+Byte', '', answer)
            texts.append([df.loc[df['url'] == url, 'name'].iloc[0],\
                          df.loc[df['url'] == url, 'field'].iloc[0],
                          question,\
                          answer])

# DataFrame으로 변환
qna_df = pd.DataFrame(texts, columns=['기업명', '산업분류', '질문', '답변'])

# csv 파일로 저장
qna_df.to_csv('qna.csv', index=False, encoding='utf-8-sig')


