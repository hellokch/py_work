# -*- coding: utf-8 -*-
#pip install selenium  
#pip install beautufulsoup4
from bs4 import BeautifulSoup
from selenium import webdriver
import requests
import time
import pandas as pd


def page_calc(tot_num):
    if tot_num % 20 == 0:
        max_page = tot_num // 20
    else:
        max_page = tot_num // 20 + 1
    return max_page

def in_data_url(start_page, end_page):
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
        
        print(i)
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
   
    # 판다스 데이터프레임으로 저장
    data = {'name': names, 'field': fields, 'url': urls}
    idxs = pd.to_numeric(idxs)
    df = pd.DataFrame(data, index = idxs)
    
    driver.quit()
    
    return df

'''
#총 이력서의 건수를 입력하세요 (처음에만)
end_page = page_calc(7563)

df_url = in_data_url(1,end_page)

df_url.to_csv('read_url_fin.csv', index=True)

len(df_url)
'''