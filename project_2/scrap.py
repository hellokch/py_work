'''
import os
import sys
import urllib.request


client_id = "9QRegot6Tc_IaARiFHbO"
client_secret = "lXpvk5gHeG"


search="삼성전자"
encText = urllib.parse.quote(search)
url = "https://openapi.naver.com/v1/search/news?query=" + encText # JSON 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
    '''




#pip install selenium  
#pip install beautufulsoup4
from bs4 import BeautifulSoup
from selenium import webdriver
import requests
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time
from urllib.parse import quote


driver = webdriver.Chrome('/chromedriver/chromedriver.exe')

name='SK하이닉스'
search_name= quote(name)

#news_office_checked=1009 : 매일경제
#news_office_checked=1015 : 한국경제
#news_office_checked=1011 : 서울경제
office = '1009'
#날짜 형식은 YYYY.mm.dd
ds = '2023.03.15'
de = '2023.03.15'
search_url = f"https://search.naver.com/search.naver?where=news&query={search_name}&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds={ds}&de={de}&docid=&related=0&mynews=1&office_type=1&office_section_code=3&news_office_checked={office}&nso=so%3Ar%2Cp%3Afrom20230313to20230314&is_sug_officeid=0"
driver.get(search_url)

print(search_url)

while True :
    response = requests.get(driver.current_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    
    elements = soup.select('.news_tit')
    
    # 요소가 존재하는 경우, 텍스트를 출력합니다.
    if elements:
        for element in elements:
            print(element.text)
            print(element.get('href')) # 요소의 href 속성값을 출력합니다.
            print('-' * 50) # 구분선을 출력합니다.
    
    next_button = driver.find_element(By.CSS_SELECTOR, '.btn_next')
    
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, '.btn_next:not([aria-disabled="true"])')
    except NoSuchElementException:
        print('No more pages.')
        break
        
    next_button.click()
    time.sleep(1)


driver.quit()




















