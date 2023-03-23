# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:19:30 2023

@author: 김충환
"""

import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")
df.info()
df.deck.unique()
#deck 컬럼의 값별 건수 출력하기
df.deck.value_counts(dropna=False)
df.deck.head()
#isnull() : 결측값? 결측값인 경우 T, 일반값F
df.deck.head().isnull()

df.deck.head().notnull() #위에꺼 반대

#결측값의 갯수 조회
df.isnull().sum()
df.isnull().sum(axis = 0)
df.isnull().sum(axis = 1)
#결측값이 아닌 갯수 조회
df.notnull().sum()
df.notnull().sum(axis = 0)
df.notnull().sum(axis = 1)
############
#dropna : 결측값 제가
#       : inplace = True
#결측값이 500개 이상인 컬럼제거
#tresh = 500 : 결측값의 갯수가 500이상
df_thresh = df.dropna(axis=1, thresh=500)
df_thresh.info()
df.info()

#결측값을 가진 행을 제거
# subset = ["age"] :컬럼설정.
# how = 'any'/'all' : 한개의 결측값/ 모든값이 결측값
# axis = 0 : 행
df_age = df.dropna(subset=["age"],how='any',axis=0)
df_age.info()
df.info()
###
# fillna : 결측값을 다른값으로 치환
#        : inplace=True가 있어야 자체 객체 변경
# fillna(치환할 값, 옵션)
# 1. age 컬럼의 값이 결측값인 경우 평균 나이로 변경하기
# 1. age 컬럼의 평균나이 조회하기
age_mean = df["age"].mean()
age_mean
age_mean = df.mean()["age"]
age_mean
#치환하기
df["age"].fillna(age_mean, inplace=True)
df.info()


#2. embark_town 컬럼의 결측값은 빈도수가 가장 많은 데이터로 치환하기
#   embark_town 중 가장 건수가 많은 값을 조회하기
#   vlaue_counts()함수 결과의 첫번째 인덱스값.
df["embark_town"].value_counts()

most_freq = df["embark_town"].value_counts().indexp[0]
most_freq
#value_counts() 함수 결과의 가장 큰 값의 인덱스값
most_freq = df["embark_town"].value_counts().idxmax()
most_freq
# embark_town 컬럼의 결측값에 most_freq 값을 치환하기
# 결측값의 인덱스 조회
df[df["embark_town"].isnull()]
df.iloc[[61,829]]["embark_town"]
df["embark_town"].fillna(most_freq,inplace=True)
df.iloc[[61,829]]["embart_town"]
df.info


# embarked 컬럼을 앞의값으로 치환하기
# 1.embarked 컬럼의 값이 결측값인 레코드 조회하기
df[df["embarked"].isnull()]
df.iloc[[61,829]]["embarked"]
df["embarked"][58,65]
df["embarked"][825,831]
#앞의 데이터로 치환하기
df["embarked"].fillna(method="ffill",inplace=True)
df["embarked"][58,65]
df["embarked"][825:831]
df.info()

#중복데이터 처리
df = pd.DataFrame({"c1" : ['a','a','b','a','b'],
                   'c2' : [1,1,1,2,2],
                   'c3' : [1,1,2,2,2]})
df
#duplicated() : 중복데이터 찾기.
#               중복된 경우 중복된 데이터 부터 True리턴
#               전체 행을 비교하여 중복 검색
df_dup = df.duplicated()
df_dup
df[df_dup] #중복데이터만 조회


#c1 컬럼을 기준으로 중복검색
col_dup = df['c1'].duplicated()
col_dup
df[col_dup]
df


mpg = pd.read_csv("data/auto-mpg.csv")
mpg.info

mpg["kpl"] = mpg["mpg"] * 0.425
mpg.info()
mpg.kpl.head()

mpg['kpl'] = mpg['kpl'].round(1)
mpg.kpl.head()
mpg.info()

mpg.horsepower.unique()





























