#pip install selenium  
#pip install beautufulsoup4
from bs4 import BeautifulSoup
from selenium import webdriver
import requests
import time
import pandas as pd
import re



def page_calc(tot_num):
    if tot_num % 20 == 0:
        max_page = tot_num // 20
    else:
        max_page = tot_num // 20 + 1
    return max_page

def init_data_url(start_page, end_page):
    #서칭된 url이 상대경로 이기때문에 앞부분 url을 따로 저장
    pre_url = 'https://www.jobkorea.co.kr'
    #드라이버 실행
    driver = webdriver.Chrome('/chromedriver/chromedriver.exe')
    
    #url 리스트 객체 초기화
    names = []
    fields = []
    urls = []
    idxs = []
    for i in range(start_page,end_page+1):
        pagenum = i
        
        #url 접근
        search_url = f'https://www.jobkorea.co.kr/starter/PassAssay?Page={pagenum}'
        driver.get(search_url)
        # web loading 대기
        time.sleep(1)
        # 드라이버 현재 URL 로드 해서 response 객체에 저장
        response = requests.get(driver.current_url)
        # 뷰티풀소프를 이용해서 html 파싱
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 웹 스크래핑한 결과에서 필요한 정보를 추출하여 각 리스트에 append
        for name in soup.select('.titTx'):
            names.append(name.text.strip())
        for field in soup.select('.field'):
            if field.text.strip() not in ['인턴', '신입']:
                fields.append(field.text.strip())
        for url in soup.find_all('p', {'class': 'tit'}):
            full_url = pre_url + url.find('a')['href']
            url_without_params = full_url.split('?')[0]
            urls.append(url_without_params)
            idx = full_url.split('View/')[1].split('?')[0]
            idxs.append(idx)
        print(f'{pagenum} 완 : {pagenum}/{end_page}')
   
    # 판다스 데이터프레임으로 저장
    data = {'name': names, 'field': fields, 'url': urls}
    idxs = pd.to_numeric(idxs)
    df = pd.DataFrame(data, index = idxs)
    
    driver.quit()
    filename = f'init_{start_page}to{end_page}'
    df.to_csv(f'data/{filename}.csv', index=True)
    print(filename + ' 저장완료')
    return filename





def read_context(file_name):
    df = pd.read_csv(f'data/{file_name}',index_col=0)
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
            texts.append([df.loc[df['url'] == url, 'name'].iloc[0],
                          df.loc[df['url'] == url, 'field'].iloc[0],
                          df.loc[df['url'] == url, 'url'].iloc[0].split('View/')[1],
                          question,
                          answer])
        cnt += 1
        print(f'{cnt} 번째 url 완')
    
    # DataFrame으로 변환
    qna_df = pd.DataFrame(texts, columns=['기업명', '산업분류','게시판번호' , '질문', '답변'])
    
    # csv 파일로 저장
    qna_df.to_csv(f'data/context_{file_name}.csv', index=False, encoding='utf-8')

    


### 처음에 해당 폴더 내 ./data/ 디렉토리를 만들어야함
#==============================
# 시작 페이지와 끝 페이지 입력 (url 불러오기)
init_data_url(1,2)

## url을 불러온 파일 이름을 실행시키면 됨
read_context('init_urls_1to2.csv')

# #총 이력서의 건수를 입력하세요 (처음에만)
# end_page = page_calc(20)


