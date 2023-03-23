# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:20:52 2023

@author: 김충환
"""

#
#머신러닝 : 기계학습, 예측 AI(인공지능)
#           변수(컬럼,피처)들의 관계를 통해서 예측하는 과정
# 지도학습  : 정답을 지정
#           회귀분석 : 가격,주가,매출, 예측하는 과정.
#                   연속성 있는 데이터의 예측에 사용
# 분류    : 데이터의 선택. 평가
# 비지도학습 : 정답이 없음.
# 군집 : 비슷한 데이터들 끼리 그룹화함.
# 강화학습 : 행동을 할 때마다 보상을 통해 학습하는 과정

# 머신러닝 프로세스
# 데이터 정리 - 데이터 분리 - 알고리즘 준비
# -> 모형학습 - 예측 - 모형평가 -> 모형활용




# 회귀분석 (regression)
# 단순회귀분석 : 독립변수, 종속변수가 한개씩
# 독립변수(설명변수) : 예측에 사용되는 데이터
# 종속변수(예측변수) : 예측해야하는 데이터

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/auto-mpg.csv")
df.info()


df["horsepower"].unique()
df["horsepower"].replace("?",np.nan,inplace=True)
df.info()
df.dropna(subset=["horsepower"],axis=0,inplace=True)
df.info()
df["horsepower"]=df["horsepower"].astype(float)
df.info()

#머신러닝에 필요한 속성(열,컬럼,변수,피처) 선택하기
ndf = df[['mpg','cylinders', 'horsepower', 'weight']]
ndf.corr()
sns.pairplot(ndf)

#독립변수, 종속변수
X=ndf[['weight']]
Y=ndf["mpg"]
len(X)
len(Y)

#train_test_split : 훈련/테스트 데이터 분리 함수
#train_test_split (독립변수, 종속변수, 테스트데이터비율, seedNum)
#test_size=0.3 : 훈련 테스트 7 3 default 0.25
#seed

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = \
    train_test_split(X,Y,test_size=0.3, random_state=10)
len(X_train)
len(X_test)
len(Y_train)
len(Y_test)



#알고리즘 준비 : 선형회귀분석
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,Y_train)

y_hat = lr.predict(X_test)

r_square = lr.score(X_test,Y_test)
r_square #결정계수
r_square = lr.score(X,Y)
r_square


#전체 데이터 평가하기
y_hat = lr.predict(X)

plt.figure(figsize=(10,5))
ax1=sns.kdeplot(Y,label="Y")
ax2=sns.kdeplot(y_hat, label="y_hat",ax=ax1)
plt.legend()
plt.show()


#알고리즘 선택 PolynomialFeatures
#LinearRegression : 선형회귀 1차
#PolynomialFeatures : 다항회귀분석 ax**2+bx+c
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
X_train.shape
X_train.iloc[0]

X_train_poly = poly.fit_transform(X_train)
X_train_poly.shape
X_train_poly[0]

#평가데이터도 다항식 데이터 변환
pr = LinearRegression()
pr.fit(X_train_poly, Y_train)

X_poly = poly.fit_transform(X)
y_hat = pr.predict(X_poly)
plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(Y, label="Y")
ax2 = sns.kdeplot(y_hat, label="y_hat", ax = ax1)
plt.legend()
plt.show

r_square = pr.score(X_poly,Y)
r_square


u = ((Y-y_hat)**2).sum()
v = ((Y-Y.mean())**2).sum()
1-(u/v)

#   단순회귀분석     : 독립변수, 종속변수가 한개인 경우
#   단항           : 1차함수
#   다항           : 다차원 함수
#   다중회귀분석     : 독립변수가 여러개, 종속변수는 한개
#   Y=a1X1 + a2X2 + anXn + b


X = ndf[['cylinders','horsepower','weight']]
Y = ndf['mpg']
X.info()


X_train, X_test, Y_train,Y_test = \
    train_test_split(X,Y,test_size=0.3,random_state=10)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, Y_train)
y_hat=lr.predict(X)

r_square = lr.score(X,Y)
r_square

plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(Y, label="Y")
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show



#####################################
#  https://data.kma.go.kr/ : 기후통계분석 > 기온분석 데이터 다운받기
#    1904 ~ 전일까지 : seoul_1215.csv 저장
#   2022-12-15일 날짜 예측하기
##############################################



seoul = pd.read_csv("data/seoul_2303.csv", encoding="cp949")
seoul.info()
seoul.head()

seoul["날짜"] = seoul["날짜"].str.replace("\t","")
seoul.info()
seoul.head()

seoul["년도"]=seoul["날짜"].str[:4]
seoul.head()

seoul["월일"]=seoul["날짜"].str[5:]
seoul.head()


seoul0306=seoul[seoul["월일"]=="03-06"]
seoul0306.info()
seoul0306.tail()

del seoul0306["지점"]

seoul0306.columns=\
    ["날짜","평균기온","최저기온","최고기온","년도","월일"]
seoul0306.info()



seoul0306[seoul0306["최저기온"].isnull()]

seoul0306 = seoul0306.dropna(subset=["최저기온"],axis=0)
seoul0306.info()

X=seoul0306[["년도"]]
Y=seoul0306["최저기온"]

model = LinearRegression()
model.fit(X,Y)
result = model.predict([[2023]])
result















