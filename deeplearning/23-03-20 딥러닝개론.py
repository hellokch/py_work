# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:28:22 2023

@author: 김충환
"""

'''
인공신경망 (ANN)
단위 : 퍼셉트론

y = x1w1 + x2w2 +b
x1,x2 : 입력값, 입력층
y : 결과값
w : 가중치
b : 편향

'''

'''
퍼셉트론을 이용하여 XOR 게이트 구현
단일신경망으로 구현안됨
다중신경망으로 구현해야함
단일퍼셉트론 : 입력측 - 출력층
다중퍼셉트론 : 입력층 - 은닉층 - 출력층
'''

import numpy as np

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

for xs in [(0,0),(0,1),(1,0),(1,1)]:
    y = XOR(xs[0], xs[1])
    print(xs, "=>", y)

'''
 Tensorflow 설치 
   1. https://www.microsoft.com/ko-kr  연결
   2. 다운로드 센터 클릭
   3. 개발자 도구 클릭
   4. 05. Visual Studio 2015용 Visual C++ 재배포 가능 패키지 클릭 
   5. 다운로드
   6. vc_redist.x64.exe 선택 => 다음클릭=> 다운받기
   7. 파일탐색기에서 vc_redist.x64.exe 실행
   
   ======================!!!!!!    annaconda 2022-10은 위의 1 ~ 7 까지 않해도 된다 
   
   8. anaconda prompt를 관리자모드로 실행
   9. pip install tensorflow
      tensorflow 버전 확인
tensorflow 1.*      
tensorflow 2.*
1,2 버전사이에 호환이 안됨.
'''
#pip install tensorflow

#버전 확인.
import tensorflow as tf
print(tf.__version__)

tf.config.list_physical_devices("GPU")



import pandas as pd
pd.__version__
import numpy as np
np.__version__

data = np.array([[0,0],[0,1],[1,0],[1,1]])
andlabel = np.array([[0],[0],[0],[1]])
orlabel = np.array([[0],[1],[1],[1]])
xorlabel = np.array([[0],[1],[1],[0]])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse

model = Sequential() #딥러닝 모델


'''
Dense : 밀집층
1 : 출력값의 갯수
input_shape : 입력값의 갯수
activation : 활성화 함수 알고리즘
linear : 선형함수
sigmoid : 0 ~ 1 사이의 값 변형
relu : 양수인 경우 선형 함수, 음수인 경우 0
'''

model.add(Dense(1, input_shape=(2,), activation='linear'))

'''
compile : 모델 설정, 모형설정. 가중치 찾는 방법 설정
optimizer = SGD() : 경사하강법 알고리즘 설정.
loss = mse : 손실함수. mse : 평균제곱오차
            mse값이 가장 적은 경우의 가중치와 편향 구함
metrics =['acc'] : 평가 방법 지졍. acc : 정확도
=> 손실함수의 값은 적은값. 정확도는 1에 가까운 가중치와 편향의 값을 찾도록 설정
'''
model.compile(optimizer=SGD(), loss=mse,metrics=['acc'])



'''
data : 훈련데이터
label : 정답
epochs = 300 : 300번 반복학습 . 손실함수가 적고, 정확도가 높아지도록
verbose = 0 : 학습과정 출력 생략
verbose = 1 : 학습과정 상세 출력 (기본값)
verbose = 2 : 학습과정 간략 출력
'''

model.fit(data,andlabel,epochs=300, verbose = 2)

print(model.get_weights())
print(model.predict(data))
print(model.evaluate(data, andlabel))

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
clf.fit(data,orlabel)

clf.predict(data)


#MNIST 데이터를 이용하여 숫자를 학습하여 숫자 인식하기
#MNIST 데이터셋 다운받기
from tensorflow.keras.datasets.mnist import load_data

(x_train, y_train), (x_test, y_test) = load_data(path='mnist.npz')
x_train.shape()
y_train.shape()
x_test.shape()
y_test.shape()

import matplotlib.pyplot as plt
import numpy as np

random_idx = np.random.randint(60000, size=3)
random_idx
for idx in random_idx :
    img = x_train[idx,:]
    label = y_train[idx]
    plt.figure()
    plt.imshow(img)
    plt.title\
        ('%d-th data, label is %d' % (idx,label), fontsize=15)


from sklearn.model_selection import train_test_split

x_train, x_val,y_train,y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=777)

x_train.shape
x_val.shape

























