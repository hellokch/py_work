# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:02:05 2023

@author: 김충환
"""

from tensorflow.keras.datasets import imdb
num_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
print(X_train.shape, X_test.shape)
print(X_train[1])
print(y_train[0])
print(len(X_train[0]))
print(len(X_train[1]))
list(imdb.get_word_index().items())[0]
list(imdb.get_word_index().items())




# 토큰인덱스의 값이 작은 경우 빈번한 단어임
# key : 단어
# value : 토큰 인덱스
imdb_get_word_index ={}
for key, value in imdb.get_word_index().items():
    imdb_get_word_index[value] = key

for i in range(1, 11):
    print('{} 번째로 가장 많이 쓰인 단어 = {}'.format(i, imdb_get_word_index[i]))


import numpy as np

lengths = np.array([len(x) for x in X_train])
lengths[:10]
X_train[0]
np.mean(lengths)
np.median(lengths)
np.max(lengths)
np.min(lengths)

import matplotlib.pyplot as plt
plt.hist(lengths)
plt.xlabel('lengths')
plt.ylabel("frequency")

y_train[:10]

# 딥러닝을 위해서는 데이터의 길이가 동일해야함
# -> 패딩작업이 필요함 : 데이터의 길이가 지정한 길이보다 작으면 0으로 채움.
#                     지정한 길이보다 크면, 지정길이로 잘라냄

#패딩방법

from tensorflow.keras.preprocessing.sequence import pad_sequences
a1 = [[1,2,3]]
a2 = [[1,2,3,4,5,6,7,8]]

# maxlen : 지정길이
# padding = 'pre' : 앞쪽을 0 으로 채움. 기본값
# padding = 'post' : 뒤쪽을 0으로 채움

a1_pre = pad_sequences(a1, maxlen=5)
a2_pre = pad_sequences(a2, maxlen=5)
print(a1_pre)
print(a2_pre)

a1_post = pad_sequences(a1, maxlen=5, padding = 'post')
print(a1_post)



max_len = 500
pad_X_train = pad_sequences(X_train, maxlen=max_len, padding="pre")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

num_words

model = Sequential()
model.add(Embedding(input_dim = num_words, output_dim=32, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam' , loss='binary_crossentropy', metrics = ['acc'])
model.summary()


history = model.fit(pad_X_train, y_train, batch_size = 32,
                    epochs = 30, validation_split = 0.2)



pad_X_test= pad_sequences(X_test, maxlen=max_len, padding='pre')
model.evaluate(pad_X_test, y_test)

pad_pre = model.predict(pad_X_test)

count = 0
for i in range(len(pad_pre)):
    if pad_pre[i] > 0.5 :
        tpre=1
    else :
        tpre=0
    if tpre == y_test[i]:
        count +=1

print(count)
print(count/len(pad_pre)*100)









plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['acc'],'g-', label='accuracy')
plt.plot(history.history['val_acc'], 'k--',label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()














import konlpy
from konlpy.tag import Okt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

train_file = tf.keras.utils.get_file('ratings_trian.text',\
                                     origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt',\
                                     extract = True)

train_file

train = pd.read_csv(train_file, sep = '\t')
train.info()
train.label.value_counts()
sns.countplot(x='label', data=train)
train.isnull().sum()
train = train.dropna()
train.info()
train.document[0]
len(train.document[0])
len(train.document[1])


p_len = train[train['label']==1]['document'].str.len()
p_len[:10]
p_len.mean()

n_len = train[train['label']==0]['document'].str.len()
n_len.mean()

#히스토그램 출력하기
fig = plt.figure(figsize = (10,5))
ax1 = plt.subplot(1,2,1)
ax1.set_title('positive')
ax1.hist(p_len)
ax2 = plt.subplot(1,2,2)
ax2.hist(n_len)
ax2.set_title('negative')
fig.suptitile("Num of characters")
plt.show()

'''
[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ] : 대문자,소문자,한글,자음,모음이 아닌 글자
    ^   : not
    A-Z : 대문자
    a-z : 소문자
    가-힣 : 한글
    ㄱ-ㅎ : 한글자음
    ㅏ-ㅣ : 한글모음
     : 공백
    
'''

train["document"].head()

train['document'] = train['document'].str.replace("[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ]","")
train["document"].head()

#불용어 제거 
okt = Okt()
def word_tokenization(text) :
    stop_words = ["는","을","를","이","가","의","던",\
                  "고","하","다","은","에","들","지","게","도"]
    return [word for word in okt.morphs(text) if word not in stop_words]


import time
start = time.time()
data = train['document'].apply((lambda x : word_tokenization(x)))
print("실행시간 : ", time.time()-start)
data.head()

data.to_csv('temp_data.txt', header=None, sep="\t")


from tensorflow.keras.preprocessing.text  import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

print("총 단어 : ", len(tokenizer.word_index))


tokenizer.word_index
tokenizer.word_counts
list(tokenizer.word_counts.items())[:10]
list(tokenizer.word_counts.values())[:10]
len(tokenizer.word_counts.values())












































