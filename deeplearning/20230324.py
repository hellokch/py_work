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
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
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

    

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--',label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--',label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7,1)
plt.legend()
plt.show()

model.evaluate(test_x, test_y)

#=============================================================
model = Sequential([
    Conv2D(input_shape=(28,28,1), kernel_size=(3,3),filters=3),
    MaxPool2D(strides=(2,2)),
    Conv2D(kernel_size=(3,3), filters=32),
    MaxPool2D(strides=(2,2)),
    Conv2D(kernel_size=(3,3), filters=64),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dropout(rate=0.3),
    Dense(units=10, activation='softmax')
    ])
model.summary()

model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=5,\
                    validation_split=0.25, batch_size=128)


    #그래프상 과적합 해소
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--',label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--',label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7,1)
plt.legend()
plt.show()

model.evaluate(test_x, test_y)



# from tensorflow.keras.preprocessing.image\
#     import load_img, img_to_array, ImageDataGenerator



train_datagen = ImageDataGenerator(\
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   shear_range=0.5,
                                   brightness_range=[0.5,1.0],
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=30,
                                   fill_mode='nearest')

image_path = tf.keras.utils.get_file\
    ('cat.jpg', 'http://bit.ly/33U6mH9')


image = plt.imread(image_path)
image.shape
image = image.reshape((1,)+image.shape)
image.shape


image
#cv2.imshow("cat",image)
#cv2.waitKey(0)



train_generator = train_datagen.flow(image, batch_size=1)

fig = plt.figure(figsize=(5,5))
fig.suptitle("augmented image")
for i in range(25):
    data = next(train_generator)
    image = data[0]
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.array(image,dtype=np.uint8))
plt.show()

########################
# 1. 캐글 데이터셋 다운받기
#    https://www.kaggle.com/trolukovich/apparel-images-dataset
# 2. archive.zip 파일을 현재 폴더의 clothes_dataset 폴더에
#    압축 풀기
# 3. 다중레이블 데이터 : 폴더의 이름이 레이블 
#      빨강  파랑   신발   드레스 
#        1    0     1      0
#        0    1     0      1
#       활성화 함수 : sigmoid
#       손실함수    : binary_crossentropy
# 4. 다중클래스 데이터 : 다중컬럼데이터
#      신발  가방  드레스 
#       1     0     0
#       0     1     0 
#       활성화 함수 : softmax
#       손실함수 : categorical_crossentropy

import numpy as np
import pandas as pd
import tensorflow as tf
import glob as glob

# glob.glob() : 파일의 목록을 리스트로 리턴
# './clothes_dataset/*/*.jpg' : 현재 폴더의 clothes_dataset폴더의 모든 하위폴다
#                             jpg 파일의 목록
# recursive = True : 지정된 폴더의 하위폴더까지 검색

all_data = np.array(glob.glob('./clothes_dataset/*/*.jpg', recursive = True))
len(all_data)
all_data.shape
all_data[:5]






