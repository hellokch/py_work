# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:05:24 2023

@author: 김충환
"""

import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import \
    accuracy_score, precision_score,recall_score,f1_score

df=pd.read_csv("data/review_data.csv")

def text_cleaning(text) :
    pattern = re.compile('[^ ㄱ-ㅎ ㅏ-ㅣ 가-힣]+')
    result = pattern.sub("", text)
    return result

def get_pos(x):
    okt = Okt()
    pos = okt.pos(x)
    pos=['{0}/{1}'.format(word,t) for word, t in pos]
    print(pos)
    return pos

data = text_cleaning(df["review"][0])

df["ko_text"] = df["review"].apply\
    (lambda x : text_cleaning(str(x)))
df.info()
df = df[df["ko_text"].str.strip().str.len() > 0]
del df["review"]

index_vectorizer=CountVectorizer\
    (tokenizer=lambda x : get_pos(x))

X = index_vectorizer.fit_transform(df["ko_text"].tolist())

# X : 독립변수, Y: 종속변수
tfidf_vectorizer = TfidfTransformer()

X = tfidf_vectorizer.fit_transform(X)

y= df["y"] #123 부정적 45 긍정적
x_train,x_test,y_train,y_test=\
    train_test_split(X,y,test_size=0.3)

lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
y_pred[:10]
y_test.values[:10]

confmat = confusion_matrix(y_test, y_pred)

print("정확도:",accuracy_score(y_test,y_pred))
print("정밀도:",precision_score(y_test,y_pred))
print("재현율:",recall_score(y_test,y_pred))
print("f1-score:",f1_score(y_test,y_pred))


#index_vectorizer : 단어들을 인덱스화 한 객체
#index_vectorizer.vocabulary_ : 딕셔너리객체
# (k(형태소 단어), v(형태소 단어의 인덱스))
#index_vectorizer.vocabulary_.items() : 딕셔너리 객체
# (k(형태소 단어의 인덱스), v(형태소 단어))
'''
invert_index_vectorizer = {v : k for k,v in\
                           index_vectorizer.vocabulary_.items()}
'''

texttoindex = {}

for tt in list(index_vectorizer.vocabulary_):
    word = tt.split("/")
    texttoindex[word[0]]=index_vectorizer.vocabulary_[tt]

indextocoef={}
for index,value in enumerate(lr.coef_[0]):
    indextocoef[index] = value



# save dictionary : indexttotext
with open('model/indextocoef.pkl','wb') as fp:
    pickle.dump(indextocoef, fp)
    print('dictionary save successfully to file')

# save dictionary : texttoindex
with open('model/texttoindex.pkl','wb') as fp:
    pickle.dump(texttoindex, fp)
    print('dictionary save successfully to file')


filename='model/lregression.rl'
pickle.dump(lr,open(filename,'wb'))


#----------------- model save end

#-------------model read
with open('model/indextocoef.pkl','rb') as fp:
    indextocoef = pickle.load(fp)
indextocoef

with open('model/texttoindex.pkl','rb') as fp:
    texttoindex = pickle.load(fp)

texttoindex

lr = pickle.load(open('model/lregression.rl','rb'))

lr.coef_[0]


