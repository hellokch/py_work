# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:41:59 2023

@author: kch0325
"""
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm

def init_data_url(start_page, end_page):
    #서칭된 url이 상대경로 이기때문에 앞부분 url을 따로 저장
    pre_url = 'https://www.jobkorea.co.kr'
    #드라이버 실행
    options = Options()
    options.headless = True  # 브라우저 창을 숨기도록 설정
    
    driver = webdriver.Chrome('/chromedriver/chromedriver.exe', options=options)
    
    #url 리스트 객체 초기화
    names = []
    fields = []
    urls = []
    idxs = []
    grades = []
    
    # URL 저장 진행도 표시
    for i in tqdm(range(start_page, end_page+1)):
    # for i in range(start_page,end_page+1):
        pagenum = i
        
        #url 접근
        search_url = f'https://www.jobkorea.co.kr/starter/PassAssay?Page={pagenum}&Pass_An_Stat=1'
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
        for grade in soup.select('.grade'): #전문가 별점 추가
            grade_text = grade.text.strip()
            grades.append(grade_text.strip())
        # print(f'{pagenum}/{end_page}')
   
    # 판다스 데이터프레임으로 저장
    data = {'name': names, 'field': fields,'url': urls, 'grades': grades }
    idxs = pd.to_numeric(idxs)
    df = pd.DataFrame(data, index = idxs)
    
    driver.quit()
    filename = f'init_{start_page}to{end_page}'
    df.to_csv(f'data/{filename}.csv', index=True)
    print(filename + ' 저장완료')
    return filename

def type_company(type_num,start_page,end_page):
    #서칭된 url이 상대경로 이기때문에 앞부분 url을 따로 저장
    schCType = type_num
    pre_url = 'https://www.jobkorea.co.kr'
    #드라이버 실행
    options = Options()
    options.headless = True  # 브라우저 창을 숨기도록 설정
    
    driver = webdriver.Chrome('/chromedriver/chromedriver.exe', options=options)
    
    #url 리스트 객체 초기화
    urls = []
    idxs = []
    grades = []
    
    # URL 저장 진행도 표시
    for i in tqdm(range(start_page, end_page+1)):
    # for i in range(start_page,end_page+1):
        pagenum = i
        
        #url 접근
        search_url = f'https://www.jobkorea.co.kr/starter/PassAssay?Page={pagenum}&Pass_An_Stat=1&schCType={schCType}'
        driver.get(search_url)
        # web loading 대기
        time.sleep(0.5)
        # 드라이버 현재 URL 로드 해서 response 객체에 저장
        response = requests.get(driver.current_url)
        # 뷰티풀소프를 이용해서 html 파싱
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 웹 스크래핑한 결과에서 필요한 정보를 추출하여 각 리스트에 append
        for url in soup.find_all('p', {'class': 'tit'}):
            full_url = pre_url + url.find('a')['href']
            url_without_params = full_url.split('?')[0]
            urls.append(url_without_params)
            idx = full_url.split('View/')[1].split('?')[0]
            idxs.append(idx)
        for grade in soup.select('.grade'): #전문가 별점 추가
            grade_text = grade.text.strip()
            grades.append(grade_text.strip())
        # print(f'{pagenum}/{end_page}')
   
    # 판다스 데이터프레임으로 저장
    data = {'url': urls, 'grades': grades }
    idxs = pd.to_numeric(idxs)
    df = pd.DataFrame(data, index = idxs)
    
    driver.quit()
    filename = f'{type_num}_{end_page}'
    df.to_csv(f'data/{filename}.csv', index=True)
    print(filename + ' 저장완료')
    return filename
# 11 ~ 18까지

def page_calc(tot_num):
    if tot_num % 20 == 0:
        max_page = tot_num // 20
    else:
        max_page = tot_num // 20 + 1
    return max_page

# type_list = {'12': 2052, '13': 2277, '14': 686, '15': 97, '16': 920, '17': 1509, '18': 405}
type_list = {'18': 405}

for ty, end in type_list.items():
    end_page = page_calc(end)
    type_company(ty, 1, end_page)





df0 = pd.read_csv('data/clean_data.csv',index_col=1)
df11 = pd.read_csv('data/11_30대그룹.csv',index_col=0)
df11_list =df11.index.tolist()

df12 = pd.read_csv('data/12_매출100.csv',index_col=0)
df12_list =df12.index.tolist()

df13 = pd.read_csv('data/13_대기업.csv',index_col=0)
df13_list =df13.index.tolist()

df14 = pd.read_csv('data/14_공공기관공사.csv',index_col=0)
df14_list =df14.index.tolist()

df15 = pd.read_csv('data/15_외국계.csv',index_col=0)
df15_list =df15.index.tolist()

df16 = pd.read_csv('data/16_중견.csv',index_col=0)
df16_list =df16.index.tolist()

df17 = pd.read_csv('data/17_주식상장.csv',index_col=0)
df17_list =df17.index.tolist()

df18 = pd.read_csv('data/18_코스닥상장.csv',index_col=0)
df18_list =df18.index.tolist()



new_df = pd.DataFrame(columns=['11'], index=df0.index)

# 인덱스 값 비교하여 데이터프레임에 값 할당
for i in df11_list:
    if i in df0.index:
        new_df.loc[i, '11'] = 1
    else:
        new_df.loc[i, '11'] = 0

print(new_df)



df_list = [    ['11', 'data/11_30대그룹.csv'],
    ['12', 'data/12_매출100.csv'],
    ['13', 'data/13_대기업.csv'],
    ['14', 'data/14_공공기관공사.csv'],
    ['15', 'data/15_외국계.csv'],
    ['16', 'data/16_중견.csv'],
    ['17', 'data/17_주식상장.csv'],
    ['18', 'data/18_코스닥상장.csv'],
]

df0 = pd.read_csv('data/clean_data.csv', index_col=1)

new_df = pd.DataFrame(columns=[i[0] for i in df_list], index=df0.index)

for df_info in df_list:
    df = pd.read_csv(df_info[1], index_col=0)
    for i in df.index:
        if i in df0.index:
            new_df.loc[i, df_info[0]] = 1
        else:
            new_df.loc[i, df_info[0]] = 0

print(new_df)
new_df.info()

'''
new_df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 3520 entries, 237371 to 144518
Data columns (total 8 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   11      1703 non-null   object
 1   12      2052 non-null   object
 2   13      2277 non-null   object
 3   14      686 non-null    object
 4   15      97 non-null     object
 5   16      920 non-null    object
 6   17      1509 non-null   object
 7   18      405 non-null    object
dtypes: object(8)
memory usage: 376.5+ KB
갯수 확인
'''

new_df.fillna(0, inplace=True)
new_df.sum()

new_df.to_csv('data/com_type.csv')
