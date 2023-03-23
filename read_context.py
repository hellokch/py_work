# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:54:28 2023

@author: kch0325
"""


import pandas as pd
'''
# csv파일을 읽고 url링크부분 리스트로 저장


df = pd.read_csv('read_url_fin.csv')
urls = df.iloc[:, -1].tolist()

# 크롤링한 텍스트들을 저장할 리스트 생성
texts = []

# 각 url링크에 접속하고 웹페이지 내용 파싱
for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    dl_con = soup.find_all('dl','qnaLists')
    for con in dl_con:
        text = con.text.split('질문Q')
        for i in range(1, len(text)):
            qna = text[i].split('보기\n\n\n답변')
            question = qna[0].replace('\n', '')
            answer = qna[1].replace('\n', '').replace('\r', '')
            texts.append(question)
            texts.append(answer)


# 질문과 답변을 저장한 리스트를 DataFrame으로 변환
df = pd.DataFrame({'question': texts[::2], 'answer': texts[1::2]})

# csv 파일로 저장
df.to_csv('qna.csv', index=False, encoding='utf-8-sig')
'''


import pandas as pd
import requests
from bs4 import BeautifulSoup

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
            texts.append([df.loc[df['url'] == url, 'name'].iloc[0], df.loc[df['url'] == url, 'field'].iloc[0], question, answer])

# DataFrame으로 변환
qna_df = pd.DataFrame(texts, columns=['기업명', '산업분류', '질문', '답변'])

# csv 파일로 저장
qna_df.to_csv('qna.csv', index=False, encoding='utf-8-sig')


