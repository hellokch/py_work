# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:22:02 2023

@author: 김충환
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib. pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
import cv2


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
red = pd.read_csv(url+'winequality-red.csv', sep=';')
white = pd.read_csv(url+'winequality-white.csv',sep=';')
red.info()
white.info

'''
1 - fixed acidity : 주석산농도
2 - volatile acidity : 아세트산농도
3 - citric acid : 구연산농도
4 - residual sugar : 잔류당분농도
5 - chlorides : 염화나트륨농도
6 - free sulfur dioxide : 유리 아황산 농도
7 - total sulfur dioxide : 총 아황산 농도
8 - density : 밀도
9 - pH : ph
10 - sulphates : 황산칼륨 농도
11 - alcohol : 알코올 도수
12 - quality (score between 0 and 10) : 와인등급
'''



#전처리
#type 컬럼 추가
#red 와인인 경우 type컬럼에 0, white와인인 경우 1을 저장
red["type"]=0
white["type"]=1

wine = pd.concat([red,white])
wine.info()
wine.head()







#wine 데이터를 minmax 정규화하여 wine_norm 데이터에 저장
wine.min()
wine.max()
wine_norm = (wine-wine.min()) / (wine.max()-wine.min())
wine_norm.head()
wine_norm.min()
wine_norm.max()

wine_shuffle = wine_norm.sample(frac=1)

wine_np = wine_shuffle.to_numpy()
type(wine_np)
wine_np.shape

train_idx = int(len(wine_np)*0.8)
train_idx

train_x, train_y = wine_np[:train_idx,:-1],wine_np[:train_idx,-1]
train_x.shape
train_y.shape
test_x,test_y = wine_np[train_idx:,:-1],wine_np[train_idx:,-1]
test_x.shape
test_y.shape


#one hot encoding

train_y = tf.keras.utils.to_categorical(train_y,num_classes=2)
test_y = tf.keras.utils.to_categorical(test_y,num_classes=2)



#모델생성
model = Sequential([
    Dense(units=48, activation='relu',input_shape=(12,)),
    Dense(units=24, activation='relu'),
    Dense(units=12, activation='relu'),
    Dense(units=2, activation='sigmoid')
    ])

model.summary()

model.compile(optimizer="adam", loss='binary_crossentropy',\
              metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=25, batch_size=32, validation_split=0.25)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--',label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--',label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7,1)
plt.legend()
plt.show()



model.evaluate(test_x, test_y)

results= model.predict(test_x)
np.argmax(results[:10],axis=-1)
np.argmax(test_y[:10],axis=-1)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(np.argmax(test_y,axis=-1),\
                     np.argmax(results,axis=-1))

plt.figure(figsize = (7,7))
sns.heatmap(cm, annot = True, fmt = 'd',cmap = 'Blues')
plt.xlabel('predicted label', fontsize =15)
plt.ylabel('true label', fontsize =15)
plt.xticks(range(2),['red','white'],rotation=45)
plt.yticks(range(2),['red','white'],rotation=0)
plt.show

#pip install opencv-python

#######################
# 컬러 이미지 분석하기
#인터넷을 통해 이미지 조회하기
#이미지 다운받기
image_path = tf.keras.utils.get_file\
    ('cat.jpg', 'http://bit.ly/33U6mH9')

image_path
image = plt.imread(image_path)
image.shape
plt.imshow(image)

image
#cv2.imshow("cat",image)
#cv2.waitKey(0)


cv2.imread(image_path)

bgr = cv2.split(image)
bgr[0].shape
bgr[1].shape
bgr[2].shape
plt.imshow(bgr[0])





titles=['RGB','Red','Green','Blue']

zero = np.zeros_like(bgr[0],dtype="uint8")
red = cv2.merge([bgr[0],zero,zero])
green = cv2.merge([zero,bgr[1],zero])
blue = cv2.merge([zero,zero,bgr[2]])


plt.imshow(red)
plt.imshow(green)
plt.imshow(blue)

images= [image,red,green,blue]

fig, axes = plt.subplots(1,4,figsize=(13,3))
objs = zip(axes, titles, images)
for ax, title, img in objs:
    ax.imshow(img)
    ax.set_title(title)
    ax.set_xticks(())
    ax.set_yticks(())



#######################################################
#  CNN : Convolutional Neural Network : 합성곱층
#        Conv2D(컨볼루션층,합성곱층) 층으로 표현
#
#  Dense 층 : 완전연결층. 밀집층
#         - 1차원 배열의 형태로 변형하여 학습.
#         - 이미지 분석시 색상의 관계 표현 못함 
#         - 3차원형태의 이미지를 1차원형태로 분석함. 
#           => 이미지 특성부분을 분석하기 어려움 
########################################################

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x,train_y),(test_x,test_y) = fashion_mnist.load_data()

train_x = train_x/255.0
test_x = test_x/255.0
train_x = train_x.reshape(-1,28,28,1)
test_x = test_x.reshape(-1,28,28,1)
train_x.shape
test_x.shape



import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
for c in range(16):
    plt.subplot(4,4,c+1)
    plt.imshow(train_x[c].reshape(28,28), cmap='gray')
plt.show()

from tensorflow.keras.layers import Conv2D,Flatten,Dense

model = Sequential([
    Conv2D(input_shape=(28,28,1), kernel_size=(3,3),filters=3),
    Conv2D(kernel_size=(3,3), filters=32),
    Conv2D(kernel_size=(3,3), filters=64),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax')
    ])

train_y[:10]
model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

train_y[0]
train_x.shape
history = model.fit(train_x, train_y, epochs=5,\
                    validation_split=0.25, batch_size=128)




















