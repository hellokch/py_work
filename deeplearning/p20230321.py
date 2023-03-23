# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:23:05 2023

@author: KITCOOP
"""

import pandas as pd
pd.__version__
import numpy as np
np.__version__
data = np.array([[0,0],[0,1],[1,0],[1,1]])
andlabel = np.array([[0],[0],[0],[1]]) #결과데이터 AND  
orlabel = np.array([[0],[1],[1],[1]]) #결과데이터 OR
xorlabel = np.array([[0],[1],[1],[0]]) #결과데이터 XOR




w=[0.5,0.5]
b=0
learning_rate = 0.7
label=xorlabel  #y  답
for  epoch in range(100) :
    # 순전파 단계: 예측값 y를 계산합니다.
    y_p = w*data + b
    
    # 손실(loss)을 계산하고 출력합니다.
    error = np.abs(y_p - label).mean()
    
    # 손실에 따른 w1, w2의 변화도를 계산하고 역전파합니다.
    w_g = learning_rate * ((y_p - label)*data).mean()
    b_g = learning_rate * (y_p - label).mean()
    
    # 경사하강법(gradient descent)를 사용하여 가중치를 갱신합니다.
    w=w-w_g
    b=b-b_g
    #print(y_p, w)
print(w)
print(b)  


def percept(x1,x2) :  #1,0
    x = np.array([x1,x2])  #입력값
    tmp = np.sum(w*x) + b
    
    #활성화 함수
    if tmp <= 0.5 :
        return 0
    else :
        return 1
    
for xs, l in zip(data, label) :
   y=percept(xs[0],xs[1])  
   print(xs,"=>",y, l)
   
   
   
   
#CNN
#1.데이터 수집
#MNIST 데이터를 이용하여 숫자를 학습하여 숫자 인식하기.
#MNIST 데이터셋 다운받기
from tensorflow.keras.datasets.mnist import load_data
(x_train, y_train),(x_test, y_test)=\
    load_data(path='mnist.npz')
x_train.shape  # (60000, 28, 28), 훈련데이터
y_train.shape  # (60000,)
x_test.shape  # (10000, 28, 28), 테스트데이터
y_test.shape  # (10000,)

x_train[0,:].shape
y_train[0]
x_test[0].shape

import matplotlib.pyplot as plt
import numpy as np
#0~59999 사이의 임의의 수 3개
random_idx = np.random.randint(60000,size=3) 
random_idx
for idx in random_idx :
    img = x_train[idx,:]
    label=y_train[idx] 
    plt.figure()
    plt.imshow(img)
    plt.title\
  ('%d-th data, label is %d' % (idx,label),fontsize=15)
  
# sample 이미지 저장  
from PIL import Image
im = Image.fromarray(x_test[0])  # shape (28,28) 
im.save("img/num.jpg")   
  
  
  
# 2. 데이터 전처리
#검증데이터 생성 : 학습 중간에 평가를 위한 데이터  
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split\
    (x_train,y_train,test_size=0.3, random_state=777)  
x_train.shape  #(42000, 28, 28)
x_val.shape    #(18000, 28, 28)


x_train[0]
y_train[0]

#데이터 정규화
'''
  MinMax normalization : X = (x-min)/(max-min)
  Robust normalization : X=(x-중간값)/(3분위값-1분위값)
  Standardization      : X=x-평균값/표준편차
'''
x_train[0]
#MinMax normalization 정규화
#현재데이터 : min:0, max=255
x_train = (x_train.reshape(42000,28*28))/255 
x_val = (x_val.reshape(18000,28*28))/255
x_test = (x_test.reshape(10000,28*28))/255
x_train[0]
x_train.shape #(42000, 784)
x_val.shape   #(18000, 784)
x_test.shape  #(10000, 784)



y_train[:10]
#레이블 전처리:one-hot 인코딩하기

from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_train[:10]
y_val=to_categorical(y_val)
y_test=to_categorical(y_test)



#3. model 학습 (CNN)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# model create
model = Sequential()  #모델 생성
model.add(Dense(64,activation="relu",input_shape=(784,)))
model.add(Dense(32,activation="relu"))
model.add(Dense(10,activation="softmax"))

# compile
model.compile(optimizer="adam", loss='categorical_crossentropy',
              metrics=['acc'])


#학습하기    history : 학습 과정을 저장한 데이터  
history=model.fit(x_train,y_train,epochs=30,batch_size=127,
                  validation_data=(x_val,y_val)) 




#4. 검증

history.history["loss"] #훈련데이터 손실함수값
history.history["acc"] #훈련데이터 정확도
history.history["val_loss"] #검증데이터 손실함수값
history.history["val_acc"] #검증데이터 정확도값
len(history.history["loss"])
type(history.history) #dict

import matplotlib.pyplot as plt
his_dict = history.history
loss = his_dict['loss']  #훈련데이터 학습시 손실함수값
val_loss = his_dict['val_loss'] #검증데이터 학습시 손실함수값
epochs = range(1, len(loss) + 1) #1 ~ 30까지의 숫자
fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(1, 2, 1) #1행2열의 1번째 그래프영역
ax1.plot(epochs, loss, color = 'blue', label = 'train_loss')
ax1.plot(epochs, val_loss, color = 'orange', label = 'val_loss')
ax1.set_title('train and val loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend() 



model.evaluate(x_test,y_test) 


# 예측하기
results = model.predict(x_test)   #x_test (10000, 784)
results[0] #7. 99%
np.argmax(results,axis=1)[0] #7
y_test[0]  #7
results[1] #2. 100%
y_test[1]  #2
np.argmax(results,axis=1)[1] #2

x_test[0,:].shape
onetest = model.predict(x_test[0,:])  #error 

#test 한개이미지 예측하기 (shape 확인)
np.reshape(x_test[0,:], (1,-1)).shape # 1d  to 2d로 수정
onetest = model.predict(np.reshape(x_test[0,:], (1,-1))) 
np.argmax(onetest,axis=1)[0] #7









#read image 예측
readimage = Image.open('img/num.jpg') 
numpyimage = np.asarray(readimage) 
numpyimage



#numpy shape setting
numpyimage.shape

#전처리 자료
normalimage=numpyimage/255 
normalimage
normalimage=normalimage.reshape(1,28*28)
normalimage.shape
#predict를 위한자료로 reshape  (1, 784)
onetest = model.predict(normalimage) 
np.argmax(onetest,axis=1)[0] #2
plt.imshow(numpyimage.reshape(28, 28)) #2차원배열. 그래프



#이미지 출력
arg_results = np.argmax(results,axis=1) #예측데이터 최대 인덱스 값들
arg_results[0]  #계산값
plt.figure(figsize=(6,6))
for idx in range(16) : #0~15까지
    plt.subplot(4, 4, idx+1)  #4행4열
    plt.axis('off') 
    plt.imshow(x_test[idx].reshape(28, 28)) #2차원배열. 그래프
    plt.title('Pred:%d,lab:%d' % \
(arg_results[idx],np.argmax(y_test[idx],axis=-1)),fontsize=15)
plt.tight_layout()
plt.show()







#예측이 틀린 이미지 16개 프린트 
count=0
for idx in range(len(results)) : #0~15까지
    number_sol = np.argmax(results,axis=1)[idx]
    number_y = np.argmax(y_test,axis=1)[idx]
    if number_y != number_sol :
        
        plt.subplot(4, 4, count+1)  #4행4열
        plt.axis('off') 
        plt.imshow(x_test[idx].reshape(28, 28)) #2차원배열. 그래프
        plt.title('Pred:%d,lab:%d' % (number_sol,number_y),fontsize=15)
  
        count +=1
        if count > 15 : break
plt.tight_layout()
plt.show()




#혼동행렬 조회하기
from sklearn.metrics import \
    classification_report,confusion_matrix
cm=confusion_matrix(np.argmax(y_test,axis=-1),\
                    np.argmax(results,axis=-1))


#heatmap으로 출력하기
import seaborn as sns
plt.figure(figsize=(7,7))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('predicted label',fontsize=15)
plt.ylabel('true label',fontsize=15)
plt.show()







import numpy as np

def softmax(arr) :
    m = np.max(arr)
    arr = arr -m
    arr = np.exp(arr)
    return arr/np.sum(arr)

def sigmoid(x) :
    return 1/(1+np.exp(-x))

case1 = np.array([3.1,3.0,2.9])
case2 = np.array([2.0,1.0,0,0.7])

print("sigmoid:",sigmoid(case1),"softmax:",softmax(case1))

# sigmoid: [0.95689275 0.95257413 0.94784644] 
# softmax: [0.3671654  0.33222499 0.30060961]
sum(softmax(case1)) #1.0
print("sigmoid:",sigmoid(case2),"softmax:",softmax(case2))
#sigmoid: [0.88079708 0.73105858 0.5        0.66818777] 
#softmax: [0.56314344 0.20716889 0.07621318 0.15347449]
sum(softmax(case2)) #1.0







##################################################
# Fashion-MNIST 데이터셋 다운로드

#1. data 수집
from tensorflow.keras.datasets.fashion_mnist import load_data
(ox_train, oy_train), (ox_test, oy_test) = load_data()
print(ox_train.shape,ox_test.shape) #(60000, 28, 28) (10000, 28, 28)
class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
oy_train[:10] #[9, 0, 0, 3, 0, 2, 7, 2, 5, 5]
ox_train[0]
plt.imshow(ox_train[8])


import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize = (5, 5))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(ox_train[i], cmap = 'gray')
    plt.xlabel(class_names[oy_train[i]])
plt.show()

#2. 데이터 전처리

#이미지 데이터 정규화
x_train = x_train/255 #minmax 정규화 (x-min)/(max-min)
x_test = x_test/255
#레이블을 onehot인코딩하기
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#검증데이터 분리. (훈련:검증)=(7:3)
x_train,x_val,y_train,y_val = \
  train_test_split(ox_train,oy_train,test_size=0.3,random_state=777)
  
  
  
  
  





