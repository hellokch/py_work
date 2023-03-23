# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 09:30:46 2023

@author: 김충환
"""
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

#pip install konlpy

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



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
df = pd.read_csv("data/review_data.csv")
df.info()

def text_cleaning(text) :
    pattern = re.compile('[^ ㄱ-ㅎ ㅏ-ㅣ 가-힣]+')
    result = pattern.sum("", text)
    return result


data = text_cleaning()
data


from sklearn.feature_extraction.text import CountVectorizer

index_vectorizer=CountVectorizer\
    (tokenizer=lambda x : get_pos(x))

X = index_vectorizer.fit_transform(['안녕하세요 케이아이씨 입니다 케이아이씨', '안녕하세요', '힘드시죠'])

index_vectorizer.vocabulary_
X

X.shape


for a in X :
    print(a[0])
    print('================')


print(X)



#------------------sample text
df["ko_text"].tolist()


SX = index_vectorizer.fit_transform(df["ko_text"].tolist())
SX
for a in SX:
    print(a[0])
    print('=============')
df["ko_text"].head()





from sklearn.feature_extraction.text import TfidfTransformer
tfidf_vectorizer = TfidfTransformer()
X = tfidf_vectorizer.fit_transform(X)
X.shape

print(X[0])





y= df["y"] #123 부정적 45 긍정적
y
df

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=\
    train_test_split(SX,y,test_size=0.3)
x_train.shape
x_test.shape


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred[:10]
y_test.values[:10]

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_test, y_pred)
confmat


from sklearn.metrics import \
    accuracy_score, precision_score,recall_score,f1_score
print("정확도:",accuracy_score(y_test,y_pred))
print("정밀도:",precision_score(y_test,y_pred))
print("재현율:",recall_score(y_test,y_pred))
print("f1-score:",f1_score(y_test,y_pred))



#각 피처 별 가중치값 조회하기
lr.coef_[0]
len(lr.coef_[0])

plt.rcParams["figure.figsize"] = [10,8]
plt.bar(range(len(lr.coef_[0])),lr.coef_[0])

sorted(((value,index) for index,value \
        in enumerate(lr.coef_[0])),reverse=True)[:5]


sorted(((value,index) for index,value \
        in enumerate(lr.coef_[0])),reverse=True)[-5:]

coef_pos_index = sorted(((value,index) \
                         for index,value in enumerate(lr.coef_[0])),\
                        reverse=True)

coef_pos_index[:5] #긍정
coef_pos_index[-5:] #부정


#index_vectorizer : 단어들을 인덱스화 한 객체
#index_vectorizer.vocabulary_ : 딕셔너리객체
# (k(형태소 단어), v(형태소 단어의 인덱스))
#index_vectorizer.vocabulary_.items() : 딕셔너리 객체
# (k(형태소 단어의 인덱스), v(형태소 단어))

invert_index_vectorizer = {v : k for k,v in\
                           index_vectorizer.vocabulary_.items()}

invert_index_vectorizer


for coef in coef_pos_index[:20] :
    print(invert_index_vectorizer[coef[1]],":",coef[1],coef[0])

for coef in coef_pos_index[-20:] :
    print(invert_index_vectorizer[coef[1]],":",coef[1],coef[0])




#명사
noun_list=[]
for coef in coef_pos_index:
    noun = invert_index_vectorizer[coef[1]].split("/")[1]
    if noun == 'Noun' :
        noun_list.append\
            ((invert_index_vectorizer[coef[1]],coef[0]))

noun_list[:10]
noun_list[-10:]

#형용사
adj_list=[]
for coef in coef_pos_index:
    adj = invert_index_vectorizer[coef[1]].split("/")[1]
    if adj == 'Adjective':
        adj_list.append\
        ((invert_index_vectorizer[coef[1]],coef[0]))
adj_list[:10]
adj_list[-10:]






#비지도 학습: 목표변수, 종속변수가 없음
# 군집 : 데이터를 그룹화

import pandas as pd
import matplotlib.pyplot as plt

#고객의 연간 구매금액을 상품 종류별로 구분한 데이터
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/\
00292/Wholesale%20customers%20data.csv'
df = pd.read_csv(uci_path, header=0)
df.info()
df.head()
X = df.iloc[:,:]
X


#데이터 정규화
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X
from sklearn import cluster

kmeans = cluster.KMeans(init='k-means++',n_clusters=5,n_init=10)


kmeans.fit(X)
cluster_label = kmeans.labels_
cluster_label
len(cluster_label)
df["cluster"]=cluster_label
df.info()
df["cluster"].unique()
df["cluster"].value_counts()

df.info()
df.groupby("cluster").mean()
df['cluster']

df.plot(kind="scatter",x="Grocery",y="Frozen",
        c="cluster", cmap="Set1", colorbar=True,
        figsize=(10,10))


df.plot(kind="scatter",x="Milk",y="Delicassen",
        c="cluster", cmap="Set1", colorbar=True,
        figsize=(10,10))


data=pd.read_csv("data/academy1.csv")


model = cluster.KMeans(init="k-means++", n_clusters=3)
model.fit(data.iloc[:,1:])
result = model.predict(data.iloc[:,1:])
result

data["group"]=result
plt.rc("font",family="Malgun Gothic")
data.plot(kind="scatter",x="국어점수",y="영어점수",
          c="group",cmap="Set1",colorbar=True,
          figsize=(7,7))

data.groupby("group").mean()


from sklearn import datasets

iris = datasets.load_iris()
iris
type(iris)
iris.data
iris.data.shape
iris.target
iris.target.shape

ttt=iris.target

labels=pd.DataFrame(iris.target)
labels.info()
labels.columns=["labels"]
labels




