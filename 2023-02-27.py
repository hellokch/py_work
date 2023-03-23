# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:24:03 2023

@author: 김충환
"""

import os.path
from bs4 import BeautifulSoup
import urllib.request as req

fp = open("data/forecast.xml", encoding='utf-8')

soup = BeautifulSoup(fp, "html.parser")
pubdate = soup.select_one("rss pubDate").string
pubdate

rss = soup.select("rss")[0]

pubdate=rss.select_one("pubDate").string
pubdate

for location in soup.select("location") :
    city = location.select_one("city").string
    wf = location.select_one("wf").string
    tmn = location.select_one("tmn").string
    tmx = location.select_one("tmx").string
    print(city,wf,tmn,tmx)


for location in soup.find_all("location"):
    city = location.find("city").string
    wf = location.find("wf").string
    tmn = location.find("tmn").string
    tmx = location.find("tmx").string
    print(city,wf,tmn,tmx)
    
url = "https://finance.naver.com/marketindex/"
res = req.urlopen(url)
soup = BeautifulSoup(res,"html.parser")

sel = lambda q : soup.select(q)
hlist = sel("div.head_info")
print(hlist)
htitle = sel("h3.h_lst")
print(htitle)


taglist = []
titlelist=[]
for tag, title in zip(hlist, htitle) :
    print(title.select_one("span.blind").string, end="\t")
    value = tag.select_one("span.value").string
    print(value, end=" ")
    change = tag.select_one("span.change").string
    print(change, end="\t")
    blinds = tag.select("span.blind")
    b = tag.select("span.blind")[0].string
    b = tag.select("span.blind")[-1].string
    print(b, end="*******\n")
    if b =="하락" :
        taglist.append(float(change) * -1)
    else :
        taglist.append(float(change))
    titlelist.append(title.select_one("span.blind").string)

print(taglist)
print(titlelist)

import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use("ggplot")
plt.rcParams['axes.unicode_minus'] = False
rc('font', family = 'Malgun Gothic')
xlab = range(len(titlelist))
plt.bar(xlab,taglist)
plt.plot(xlab,taglist)
plt.xticks(xlab,titlelist,rotation='vertical')




#pip install selenium
from selenium import webdriver
import time

#브라우저 실행
driver = webdriver.Chrome("temp/chromedriver.exe")
driver.get("http://python.org")
time.sleep(1)

# find_elements : 선택된 태그 들
# css selector : css 언어에서 사용하는 선택자 방식
# #top ul.menu li : id="top" 인 태그의 하위 태그중
#                   class="munu" 인 ul 태그. ul 태그의 하위 li 태그들


menus = driver.find_elements("css selector","#top ul.menu li")
menus[0].text
menus[1].text
for m in menus :
    if m.text == "PyPI":
        pypi = m
        print(m.tag_name,m.text)

pypi.click()
time.sleep(5)
driver.quit()

'''
execute_script() : 자바스크립트 함수 실행.
       document.getElementsByName('id')[0].value='aaa'
       document.getElementsByName("id") : name 속성이 id인 태그 들
find_elements(선택방법,선택자) : 여러개 태그 선택 => selenium 4버전이후
find_element(선택방법,선택자) : 하나의 태그 선택
       xpath : xml방식으로 태그 찾아가는 방법
             //*[@id="log.login"] : id속성값이 log.login 인 태그 한개 선택.
                 // : root 노드. 최상위 노드. 처음부터
                 *  : 모든태그
                 [] : 속성값
                 @id : id속성
                 log.login : id속성 값
                 
  선택방법에 사용되는 문자열
   By.XPATH        : "xpath" : xml에서 사용되는 태그 검색 방식
   By.CLASS_NAME   : "class name" : class 속성값
   By.CSS_SELECTOR : "css selector" : css 문서에서 사용하는 선택자 방식      
   By.ID           : "id"    : id 속성값
   By.TAG_NAME     : "tag name" : 태그이름      
'''

from selenium import webdriver
from selenium.webdriver.common.by import By
import time
driver = webdriver.Chrome("temp/chromedriver")
driver.get("https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com")


#input : 화면에서 문자열 입력
id = input("네이버 아이디를 입력하세요 :")
#아이디 칸에 입력된 네이버 아이디 전달
driver.execute_script("document.getElementsByName('id')[0].value='"+id+"'")
time.sleep(1)
pw = input("비밀번호 :")
driver.execute_script("document.getElementsByName('pw')[0].value'"+pw+"'")
time.sleep(1)



import urllib.request as req
import os


driver = webdriver.Chrome("temp/chromedriver")
driver.get("https://search.daum.net/search?w=img&nil_search=btn&DA=NTB&enc=utf8&q=%EC%9E%A5%EB%AF%B8")
images = driver.find_elements("css selector", "#imgList > div > a > img")
img_url =[]
images
len(images)
for image in images :
    url = image.get_attribute('src')
    img_url.append(url)
print(img_url)
driver.quit()

img_folder = './img'
if not os.path.isdir(img_folder):
    os.mkdir(img_folder)
for index, link in enumerate(img_url):
    req.urlretrieve(link, f'./img/{index}.jpg')









import pandas as pd
chipo = pd.read_csv("data/chipotle.tsv", sep="\t")
chipo




























