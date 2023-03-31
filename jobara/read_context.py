
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup



df = pd.read_csv('data/read_url_20.csv',index_col=0)
# texts 리스트선언
cnt = 0
texts = []
# df데이터프레임 url컬럼만큼 반복. 
for url in df['url']:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser') #파싱
    dl_con = soup.find('dl','qnaLists') # 'dl'태그의 'qnaLists'클래스를 탐색 후 dl_con에 저장
    text = dl_con.text.split('질문Q')
    for i in range(1, len(text)):
        qna = text[i].split('보기\n\n\n답변')
        question = qna[0].replace('\n', '')
        answer = qna[1].replace('\n', '').replace('\r', '')
        # 글자수 정보 제거하기#
        answer = re.sub(r'글자수\s[\d,]+자[\d,]+Byte', '', answer)
        #texts 객체에 data frame으로 저장할 데이터 할당 #
        texts.append([df.loc[df['url'] == url, 'name'].iloc[0],\
                      df.loc[df['url'] == url, 'field'].iloc[0],
                      question,\
                      answer])
    cnt += 1
    print(f'{cnt} 번째 url 완')

# DataFrame으로 변환
qna_df = pd.DataFrame(texts, columns=['기업명', '산업분류', '질문', '답변'])

# csv 파일로 저장
qna_df.to_csv('data/read_context_20.csv', index=False, encoding='utf-8')