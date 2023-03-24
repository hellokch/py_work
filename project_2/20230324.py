# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:00:56 2023

@author: kch0325
"""
#pip install --upgrade pip
#pip install tensorflow
#pip install tensorflow-gpu
#python -c "import tensorflow as tf; print(tf.__version__)"
#pip install opencv-python
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



