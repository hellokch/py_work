# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:06:56 2023

@author: KITCOOP
"""

'''
    titanic 속성
pclass : Passenger Class, 승객 등급
survived : 생존 여부
name : 승객 이름
sex : 승객 성별
age : 승객 나이
sibsp : 탑승 한 형제/배우자 수
parch : 탑승 한 부모/자녀 수
ticket : 티켓 번호
fare : 승객 지불 요금
cabin : 선실 이름
embarked : 승선항 (C = 쉘 부르그, Q = 퀸즈타운, S = 사우스 햄튼)
body : 사망자 확인 번호
home.dest : 고향/목적지
'''
import pandas as pd
df_train = pd.read_csv("data/titanic_train.csv")
df_test = pd.read_csv("data/titanic_test.csv")
df_train.info()
df_test.info()
# df_train 데이터에서 생존여부를 그래프로 출력하기
df_train["survived"].value_counts()
df_train["survived"].value_counts().plot.bar()
df_train["survived"].value_counts().plot(kind="bar")

# 좌석 등급별 생존여부 조회하기
df_train.groupby("pclass")["survived"].value_counts()
df_train[["pclass","survived"]].value_counts()






#index로 정렬
df_train.groupby("pclass")["survived"].\
    value_counts().sort_index()
df_train[["pclass","survived"]].value_counts().\
    sort_index()
#그래프 출력
df_train[["pclass","survived"]].value_counts().\
    sort_index().plot(kind="bar")






# df_train 데이터에서 pclass별 건수를 그래프    
#   hue="survived" : 건수를 survived 컬럼의 값을 분리
#   y축의값 : 건수. 

import seaborn as sns
sns.countplot(x="pclass",hue="survived",data=df_train)
df_train.info()
df_test.info()


# 1. age 컬럼의 결측값을 
# df_train데이터의 평균값으로 변경하기 (df_train,df_test)
age_mean = df_train["age"].mean()
age_mean
df_train["age"]=df_train["age"].fillna(age_mean)
df_test["age"]=df_test["age"].fillna(age_mean)
df_train.info()
df_test.info()


#2. embarked 컬럼의 결측값을 최빈값으로 변경
embarked_freq = df_train["embarked"].value_counts().idxmax()
embarked_freq
df_train["embarked"]=df_train["embarked"].fillna(embarked_freq)
df_train.info()



#3. name,ticket,cabin,body,home.dest 컬럼 제거하기
df_train = df_train.drop\
    (["name","ticket","cabin","body","home.dest"], axis=1)
df_test = df_test.drop\
    (["name","ticket","cabin","body","home.dest"], axis=1)    


df_train.info()
df_test.info()
#4. df_train,df_test 데이터를 통합하기.
whole_df = df_train.append(df_test)   
whole_df.info()


#훈련데이터의 갯수 => 훈련데이터/테스트 데이터 분리
train_num = len(df_train)
train_num

#원핫코딩(범주형 변환)
whole_df_encoded = pd.get_dummies(whole_df)
whole_df_encoded.info()

#5. 설명변수 목표변수 분리
df_train = whole_df_encoded[:train_num]
df_test = whole_df_encoded[train_num:]
#훈련 설명변수
x_train = df_train.loc \
    [:,df_train.columns != "survived"].values
x_train.shape
x_test = df_test.loc \
    [:,df_test.columns != "survived"].values
x_train.shape

#훈련 목표변수
y_train = df_train["survived"].values
y_test = df_test.loc \
    [:,df_test.columns != "survived"].values


#로지스틱 회귀분석 모델을 이용하여 분류하기
#로지스틱 회귀분석 분류 알고리즘.
#0~1사이의 값을 리턴.
#0.5미만 :0,0.5이상 :1
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred[:10]
y_test[:10]


from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(y_test, y_pred)
con_mat

from sklearn.metrics import accuracy_score,\
    recall_score,precision_score,f1_score
print("정확도:",accuracy_score(y_test,y_pred))
print("정밀도:",precision_score(y_test,y_pred))
print("재현율:",recall_score(y_test,y_pred))
print("f1-score:",f1_score(y_test,y_pred))















################ cabin 이용 분석
import pandas as pd
df_train = pd.read_csv("data/titanic_train.csv")
df_test = pd.read_csv("data/titanic_test.csv")
df_train.info()
df_test.info()




import pandas as pd
df_train = pd.read_csv("data/titanic_train.csv")
df_test = pd.read_csv("data/titanic_test.csv")
df_train.info()
df_test.info()

# 1. age 컬럼의 결측값을 
# df_train데이터의 평균값으로 변경하기 (df_train,df_test)
age_mean = df_train["age"].mean()
age_mean
df_train["age"]=df_train["age"].fillna(age_mean)
df_test["age"]=df_test["age"].fillna(age_mean)
df_train.info()
df_test.info()

#2. embarked 컬럼의 결측값을 최빈값으로 변경
embarked_freq = df_train["embarked"].value_counts().idxmax()
embarked_freq
df_train["embarked"]=df_train["embarked"].fillna(embarked_freq)
df_train.info()

#4. df_train,df_test 데이터를 통합하기.
whole_df = df_train.append(df_test)   
whole_df.info()
whole_df


whole_df["cabin"].unique()
'''
E36 => E
B96 B98 => B
cabin 컬럼의 첫번째 문자만 추출하여 cabin 컬럼에 저장
'''
whole_df["cabin"] = whole_df["cabin"].str[0]
whole_df["cabin"].unique()




whole_df["cabin"] = whole_df["cabin"].fillna("X")
whole_df["cabin"].value_counts()

whole_df["cabin"] = whole_df["cabin"].replace({"G":"X","T":"X"})
whole_df["cabin"].value_counts()

sns.countplot(x="cabin", hue="survived", data=whole_df)



whole_df["name"].head()

name_grade=whole_df["name"].apply \
    (lambda x:x.split(", ")[1].split(".")[0])
name_grade.unique()
name_grade.value_counts()

grade_dict = {
 "A":["Rev","Col","Major","Dr","Capt","Sir"], #명예직
 "B":["Ms","Mme","Mrs","Dona"],  #여성
 "C":["Jonkheer","the Countess"], #귀족
 "D":["Mr","Don"],               #남성
 "E":["Master"],                 #젊은 남성
 "F":["Miss","Mlle","Lady"]      #젊은 여성
  }

def give_grade(g):
    for k,v in grade_dict.items():
        for title in v:
            if g == title :
                return k
    return 'G'

name_grade = \
    list(name_grade.map(lambda x : give_grade(x)))
name_grade[:10]

sns.countplot(x="name", hue="survived", data = whole_df)
whole_df.info()
whole_df = whole_df.drop\
    (["ticket", "body", "home.dest"], axis=1)
whole_df.info()

whole_df.head()


df_train = whole_df_encoded[:train_num]
df_test = whole_df_encoded[train_num:]
#훈련 설명변수
x_train = df_train.loc \
    [:,df_train.columns != "survived"].values
x_train.shape
x_test = df_test.loc \
    [:,df_test.columns != "survived"].values
x_train.shape


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred[:10]
y_test[:10]


from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(y_test, y_pred)
con_mat

from sklearn.metrics import accuracy_score,\
    recall_score,precision_score,f1_score
print("정확도:",accuracy_score(y_test,y_pred))
print("정밀도:",precision_score(y_test,y_pred))
print("재현율:",recall_score(y_test,y_pred))
print("f1-score:",f1_score(y_test,y_pred))



#pip install konlpy

#한글분석을 위한 모듈 : konlpy

# 형태소 분석 모듈
# okt(open korean Text)
# KKma(코코마)
# Komoran(코모란)
# Hannanum(한나눔)

from konlpy.tag import Okt,Kkma,Komoran,Hannanum
import time

okt=Okt()
kkma=Kkma()
komoran=Komoran()
han=Hannanum()

def sample_ko_pos(text):
    print(f"===={text} ====")
    start = time.time()
    #pos(text) : text를 형태소를 분리. 품사 표시
    print("kkma:",kkma.pos(text),"실행시간 :",time.time()-start)
    print("\n")
    start = time.time()
    print("komoran:",komoran.pos(text),"실행시간 :",time.time()-start)
    print("\n")
    start = time.time()
    print("okt:",okt.pos(text),"실행시간 :",time.time()-start)
    print("\n")
    start = time.time()
    print("han:",han.pos(text),"실행시간 :",time.time()-start)
    print("\n")

text1 = "영실아 안녕 오늘 날씨 어때?"
sample_ko_pos(text1)

text2 = "영실아안녕오늘날씨어때?"
sample_ko_pos(text2)

text3 = "안녕 ㅎㅏㅅㅔ요 ㅈㅓ는 ㄷㅐ 학생입니다."
sample_ko_pos(text3)

#pip install selenium

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import re
import time
path = 'temp/chromedriver.exe'
source_url = "https://map.kakao.com/"
driver = webdriver.Chrome(path)
driver.get(source_url)
time.sleep(1)

searchbox = driver.find_element(By.ID,"search.keyword.query")
searchbox.send_keys("강남역 고기집")
time.sleep(1)

searchbutton=driver.find_element \
    (By.ID,"search.keyword.submit")
#arguments[0].click() : searchbutton 클릭
driver.execute_script("arguments[0].click();", searchbutton)
time.sleep(1)



html = driver.page_source

soup = BeautifulSoup(html, "html.parser")
moreviews = soup.find_all \
    (name="a", attrs={"class":"moreview"})
page_urls = []
for moreview in moreviews:
    page_url = moreview.get("href")
    page_urls.append(page_url)
driver.close()
print(page_urls)
print(len(page_urls))

page='https://place.map.kakao.com/95713992'
# 필요한 컬럼명을 지정하여 빈 데이터프레임 생성
columns = ['score', 'review']
df = pd.DataFrame(columns=columns)

# 크롬드라이버 실행
driver = webdriver.Chrome(path)

# 크롤링할 페이지 접속
driver.get(page)

# 로딩을 위한 대기 시간 설정
time.sleep(2)

# '후기 더보기' 버튼이 존재하는 경우 계속 클릭하여 후기 로딩
for i in range(11):
    time.sleep(2)
    another_reviews = driver.find_element(By.CSS_SELECTOR,"span.txt_more")
    try :
        # '후기 더보기' 버튼 클릭
        another_reviews.text.index('후기 더보기')
        another_reviews.click()
    except :
        # 더 이상 로딩할 후기가 없는 경우 종료
        break

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
contents_div = soup.find(name = 'div', attrs={"class":"evaluation_review"})
reviews = contents_div.find_all(name="p", attrs={"class":"txt_comment"})
rates= contents_div.find_all(name="span",attrs={"class":"inner_star"})

rates
reviews


for rate, review in zip(rates, reviews):
    rate_style = rate.attrs['style'].split(":")[1]
    rate_text=int(rate_style.split("%")[0])/20
    row =[int(rate_text), review.find(name="span").text]
    series = pd.Series(row, index=df.columns)
    df = df.append(series, ignore_index=True)

driver.close()
df.info()
df.head()
df.score.value_counts()


'''
# 페이지 소스 가져오기
html = driver.page_source

# BeautifulSoup으로 파싱
soup = BeautifulSoup(html, 'html.parser')

# 리뷰와 별점을 담을 리스트 초기화
reviews = []
rates = []

# 리뷰와 별점이 포함된 div 태그 찾기
contents_div = soup.find(name='div', attrs={"class": "evaluation_review"})

# 리뷰와 별점 추출
reviews = contents_div.find_all(name="p", attrs={"class": "txt_comment"})
rates = contents_div.find_all(name="span", attrs={"class": "inner_star"})

# 추출한 리뷰와 별점을 DataFrame에 추가
for rate, review in zip(rates, reviews):
    rate_style = rate.attrs['style'].split(":")[1]
    rate_text = int(rate_style.split("%")[0]) / 20
    row = [int(rate_text), review.find(name="span").text]
    series = pd.Series(row, index=df.columns)
    df = df.append(series, ignore_index=True)

# 드라이버 종료
driver.close()

# 데이터프레임 정보 출력 및 일부 데이터 확인
df.info()
df.head()

# score 열 값들의 빈도수 출력
df.score.value_counts()

'''

#별점 5,4 => 긍정(1)
#별점 1,2,3 => 부정(0)

df["y"] = df["score"].apply\
    (lambda x : 1 if float(x)>3 else 0)
df.info()
df["y"].value_counts()
#df 데이터를 review_data.csv 파일로 저장
df.to_csv("data/review_data.csv", index=False)


#data/review_data.csv 읽어서 df에 저장하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time

df=pd.read_csv("data/review_data.csv")
df.info()


def text_cleaning(text):
    nhangul = re.compile('[^ ㄱ-ㅣ 가-힣]+')
    result = nhangul.sub("",text)
    return result



'''
[^ ㄱ-ㅣ 가-힣]+ : 공백, 한글이 아닌경우
^ : not
ㄱ-ㅣ (모음): 자음,모음 한개
가-힣 : 가 부터 힣 한글
+ : 한개 이상
'''

data = text_cleaning("!!!***가나다 123 라마사아 ㅋㅋㅋ 123 fff")
data

#리뷰에서 한글과 공백만 남김.
df["ko_text"] = df["review"].apply\
    (lambda x : text_cleaning(str(x)))
df.info()

df = df[df["ko_text"].str.strip().str.len() > 0]
df.info()
df.head()
df.review.head()


del df["review"]

from konlpy.tag import Okt

def get_pos(x):
    okt = Okt()
    pos = okt.pos(x)
    #컴프리헨션 방식 리스트 객체
    pos=['{0}/{1}'.format(word,t) for word, t in pos]
    print(pos)
    return pos

result = get_pos(df["ko_text"].values[0])
result

df.head()
df.info()

#글뭉치 변환하기 : 단어들을 인덱스호
from sklearn.feature_extraction.text import CountVectorizer
#글뭉치 : 분석을 위한 글모임
index_vectorizer=CountVectorizer\
    (tokenizer=lambda x : get_pos(x))

X = index_vectorizer.fit_transform(df["ko_text"].tolist())
X.shape
for a in X[0]:
    print(a)





