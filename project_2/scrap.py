#pip install selenium  
#pip install beautufulsoup4
from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.parse import quote
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import requests
import time
import pandas as pd
import re


def format_date(date_str):
    regex = r"(\d{4})(\d{2})(\d{2})"
    subst = r"\1.\2.\3"
    result = re.sub(regex, subst, date_str)
    return str(result)

#(name, in_data는 'YYYYMMDD'로 입력해야함)
def scrap_url(in_name, in_date):
    
    name = str(in_name)
    search_name= quote(name)
    date=format_date(in_date)
    #날짜 형식은 YYYY.mm.dd
    ds = date
    de = date    
    #news_office_checked=1009 : 매일경제
    #news_office_checked=1015 : 한국경제
    #news_office_checked=1011 : 서울경제
    office_list = ['1009','1011','1015']
    news_list = []
    driver = webdriver.Chrome('/chromedriver/chromedriver.exe')
    for office in office_list:
        search_url = f"https://search.naver.com/search.naver?where=news&query={search_name}&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds={ds}&de={de}&docid=&related=0&mynews=1&office_type=1&office_section_code=3&news_office_checked={office}&nso=so%3Ar%2Cp%3Afrom20230313to20230314&is_sug_officeid=0"
        driver.get(search_url)
        
        if "에 대한 검색결과가 없습니다." in driver.page_source:
            print(f'{name}에 대한 검색결과가 없습니다.')
            continue
        
        while True :
            response = requests.get(driver.current_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            elements = soup.select('.news_tit')
            # 요소가 존재하는 경우, 텍스트를 출력합니다.
            if elements:
                for element in elements:
                    #if name in element.text:
                        news_dict = {}
                        news_dict['종목명'] = name
                        news_dict['date'] = de
                        news_dict['신문사'] = office
                        news_dict['title'] = element.text
                        news_dict['url'] = element.get('href')
                        
                        news_list.append(news_dict)
            next_button = driver.find_element(By.CSS_SELECTOR, '.btn_next')
            
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, '.btn_next:not([aria-disabled="true"])')
            except NoSuchElementException:
                print(f'{office} 끝')
                break
                
            next_button.click()
            time.sleep(1)
    
    driver.quit()
    
    df = pd.DataFrame(news_list)
    df = df.drop_duplicates()
    df.to_csv(f'data/{name}_{date}.csv', index=False)
    print(f'data/{name}_{date}.csv 저장')




scrap_url("카카오페이","20230311")





