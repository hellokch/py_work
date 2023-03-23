# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 20:57:49 2023

@author: gocu9778
"""
import requests
from bs4 import BeautifulSoup

# csv파일을 읽고 url링크부분 리스트로 저장
urls = []
with open('read_url_fin.csv', 'r' , encoding='UTF8') as f:
    rows = f.readlines()
    for row in rows[1:]: # 첫 줄 제외하고 읽기
        row = row.strip() # 공백과 개행 제거
        url = row.split(',')[-1] # 컬럼쉼표구분, 마지막 컬럼만 선택하기
        urls.append(url)

# 크롤링한 텍스트들을 저장할 리스트 생성
texts = []

# 각 url링크에 접속하고 웹페이지 내용 파싱
for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 원하는 텍스트 추출하기 (예: '' 태그의 텍스트) 
    try:
        text = soup.find_all('div', 'tx') #div tx로 했을때 첫 div tx만 긁어와지는 문제 발생. find_all로 할시 아예 안 긁어와짐.
        
        for t in text :
            print(t.text)
            texts.append(t.text)
        
        
    except AttributeError: # 에러가 발생하면 
        texts.append('') # 빈 문자열 추가
    print("*******************************")

# 파일 저장 (''이름으로저장)






len(texts)



'''
parsed_text = "파싱된 텍스트\n\r입니다."
cleaned_text = parsed_text.replace('\n', '').replace('\r', '')
print(cleaned_text)  # "파싱된 텍스트입니다."
'''

dl_con.text
soup = BeautifulSoup(response.text, 'html.parser')
dl_con = soup.find('dl','qnaLists')
q_list = dl_con.text.split('질문Q')


len(q_list)

with open('crawlingtest44.txt', 'w', encoding='utf-8') as f:
    for text in texts:
        f.write(text + '\n') # 개행 추가