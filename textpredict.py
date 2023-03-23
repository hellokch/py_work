# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:15:20 2023

@author: 김충환
"""
import pickle
import pandas as pd
from textmodelsave import text_cleaning,get_pos

#--------- 저장된 모델 불러오기
with open('model/indextocoef.pkl','rb') as fp:
    indextocoef = pickle.load(fp)
indextocoef

with open('model/texttoindex.pkl','rb') as fp:
    texttoindex = pickle.load(fp)
texttoindex

lr = pickle.load(open('model/lregression.rl','rb'))

lr.coef_[0]
#-------------- load end

df=pd.read_csv("data/review_data.csv")

review=df["review"][11]
review

y = df["y"][11]
text=text_cleaning(review)
text_okt = get_pos(text)
text_okt



sol=0
for tt in text_okt :
    print(tt)
    word=tt.split("/")
    cofindex = texttoindex[word[0]]
    print(word[0],":",cofindex,":",indextocoef[cofindex])
    sol += indextocoef[cofindex]

if sol >= 0 :
    print(text,"=",y,"\n========긍정글입니다.")
else :
    print(text,"=",y,"\n========긍정글입니다.")











































