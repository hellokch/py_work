import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 


df = pd.read_csv('data/clean_data.csv', index_col=0)



df["기업명"]

CJ_df = df[df['기업명'].str.contains('CJ|씨제이')] 
GS_df = df[df['기업명'].str.contains('GS')] 
NH_df = df[df['기업명'].str.contains('NH|농협')] 
SK_df = df[df['기업명'].str.contains('SK|에스케이')] 
SS_df = df[df['기업명'].str.contains('삼성')] 
HJ_df = df[df['기업명'].str.contains('한국전력')] 
ETC_df = df[~df['기업명'].str.contains('CJ|씨제이|GS|NH|농협|SK|에스케이|삼성|한국전력')]
CJ_df.to_csv("data/CJ_df.csv", index=False)
GS_df.to_csv("data/GS_df.csv", index=False)
NH_df.to_csv("data/NH_df.csv", index=False)
SK_df.to_csv("data/SK_df.csv", index=False)
SS_df.to_csv("data/SS_df.csv", index=False)
HJ_df.to_csv("data/HJ_df.csv", index=False)
ETC_df.to_csv("data/ETC_df.csv", index=False)



df_list = ['CJ_df', 'GS_df', 'NH_df', 'SK_df', 'SS_df', 'HJ_df', 'ETC_df']

for df_name in df_list:
    df = pd.read_csv(f'data/{df_name}.csv')
    a = df["등급"] + df["점수"]
    company = df_name.split("_")[0]
    print(company)

    a = CJ_df["등급"] + CJ_df["점수"]
    
    
    a_list = a.tolist()
    np.median(a_list) # -2.0
    
    threshold = np.median(a_list)
    
    greater = sum(1 for x in a if x > threshold) #60
    less = sum(1 for x in a if x <= threshold) #61
    
    # classify 컬럼 초기화
    a_df = pd.DataFrame(a, columns=['a'])
    a_df['classify'] = 0
    a_df['answer']=CJ_df["답변"]
    # classify 컬럼 값 설정
    a_df.loc[a_df['a'] > np.median(a_list), 'classify'] = 1
    
    # 결과 확인
    print(a_df)
    
    
    
    # 데이터 프레임에서 텍스트 데이터만 추출하여 리스트로 변환
    corpus = a_df['answer'].tolist()
    
    # CountVectorizer 객체 생성
    vectorizer = CountVectorizer()
    
    # 단어-문서 행렬 생성
    X = vectorizer.fit_transform(corpus)
    X.shape
    
    #2.가중치 확인
    #X: 독립변수,  y: 종속변수 (1이상 긍정/0,음수 값 부정)
    tfidf_vectorizer = TfidfTransformer() 
    X = tfidf_vectorizer.fit_transform(X) #가중치만들기
    y=a_df["classify"]  #0: 부정, 1:긍정
    
    #훈련데이터,테스트 데이터로 분리
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=3)
    
    #Logistic Regression 알고리즘을 이용하여 분류
    
    lr=LogisticRegression(random_state=0) #고정된 난수 생성
    lr.fit(x_train, y_train)
    
    y_pred=lr.predict(x_test)
    y_pred[:10]  #예측데이터
    
    
    # 평가하기 -- 값 확인해보기
    confmat = confusion_matrix(y_test, y_pred)
    print(f"{company} - confusion matrix:")
    print(confmat)
    
    
    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{company} - accuracy(정확도): {accuracy}")
























