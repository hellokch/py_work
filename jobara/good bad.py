# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:01:24 2023

@author: kch0325
"""
import requests
import time
import pandas as pd
import re
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm



# URL 파일 읽어오기
file_name = 'init_1to2.csv'
in_df = pd.read_csv(f'data/{file_name}',index_col=0)
# texts 리스트선언
Q = []
A = []
P = []
name = []
field = []
board_num = []
for url in tqdm(in_df['url']):
    # df데이터프레임 url컬럼만큼 반복. 
    try:
        response = requests.get(url)
        response.raise_for_status()
    except (requests.exceptions.HTTPError, requests.exceptions.RequestException):
        print(f'{url}에서 오류 발생. 다음 URL로 넘어갑니다.')
        continue
    soup = BeautifulSoup(response.text, 'html.parser')
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # 메뉴 text제외하고 받아오기
    que_div_txs = soup.find_all('span', 'tx')[12:]
    ans_div_txs = soup.find_all('div', 'tx')[:len(que_div_txs)]
       
    # dl_con이 None인 경우 처리 추가
    if ans_div_txs is None:
        print(f'{url}에서 ans_div_txs을 찾을 수 없습니다. 다음 URL로 넘어갑니다.')
        continue
    # 질문 추출 (col 구분이 ,이므로 모두 제거)
    for que_div_tx in que_div_txs:
        Q.append(que_div_tx.text.replace(",","").replace('\n', '').replace('\r', ''))
    
    # 답변 추출
    for ans_div_tx in ans_div_txs:
        points = 0
        # Qn 번째 답변에서 b 태그가 goot 이면 +1 bad면 -1
        for b_tag in ans_div_tx.find_all('b'):
            if 'good' in b_tag['class']:
                points += 1
            elif 'bad' in b_tag['class']:
                points -= 1
        
        #파싱된 답변 데이터에서 span 태그와 'p'태그를 제외
        for tag in ans_div_tx('span'):
            tag.decompose()
        for tag in ans_div_tx('p'):
            tag.decompose()
        # 텍스트 추출 (col 구분이 ,이므로 모두 제거)
        A.append(ans_div_tx.get_text(strip=True).replace(",","").replace('\n', '').replace('\r', ''))
        P.append(points)
        # in_df 의 정보 추가
        # 기업명
        name.append(in_df.loc[in_df['url'] == url, 'name'].iloc[0])
        # 산업분류
        field.append(in_df.loc[in_df['url'] == url, 'field'].iloc[0])
        # 게시판번호
        board_num.append(in_df.loc[in_df['url'] == url, 'url'].iloc[0].split('View/')[1])


# DataFrame으로 변환
out_df = pd.DataFrame({"기업명":name , "산업분류":field, "게시판번호":board_num, "질문":Q, "답변":A, "점수":P})
out_df.to_csv(f'data/context_{file_name}', index=False, encoding='utf-8')



















