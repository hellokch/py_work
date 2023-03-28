# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:35:43 2023

@author: 김충환
"""

import pickle
import pandas as pd
import matplotlib. pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = models.load_model('model/cloths_model1.h5')
history = pickle.load(open('model/cloths_model_hist', "rb"))
test_df = pd.read_csv("./clothes_dataset/test.csv")
test_df.info()

class_col = list(test_df.columns[1:])
class_col
batch_size = 32

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history['loss'], 'b-', label='loss')
plt.plot(history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history['acc'],'g-', label='accuracy')
plt.plot(history['val_acc'], 'k--',label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# 테스트 데이터를 이용하여 예측하기
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe = test_df,
    directory = None,
    x_col = 'image',
    y_col = None,
    target_size = (112, 112),
    color_mode = 'rgb',
    class_mode = None,
    batch_size = batch_size,
    shuffle = False)


preds = model.predict(test_generator, steps = 32)
preds[0]

timage = plt.imread(test_df["image"][0])
plt.imshow(timage)
preds[0]
class_col[8]

off = 8
do_preds = preds[off : off+8] #8개씩 이미지 조회


for i, pred in enumerate(do_preds):
    plt.subplot(2, 4, i + 1)
    prob = zip(class_col, list(pred))
    prob = sorted(list(prob),\
                  key = lambda z : z[1], reverse = True)[:2]
    image = plt.imread(test_df["image"][i+off])
    plt.imshow(image)
    plt.title(f'{prob[0][0]}:\
{round(prob[0][1]*100,2)}%\
\n{prob[1][0]}:\
{round(prob[1][1]*100,2)}%')
    plt.tight_layout()
plt.show()



# RNN (Recurrent Neural Network) : 순환신경망
# 음성인식, 문장번역 등에 사용

# SimpleRNSubplot
import numpy as np
# return_sequences = True : 순환결정
# activation = 'tanh' : -1 ~ 1 사이의 값을 가짐
rnn1 = SimpleRNN(units = 1, activation = 'tanh', return_sequences=True)

X = []
Y = []


for i in range(6) : 
    # lst = list[0.1,0.2,0.3,0.4]
    lst = list(range(i, i+4))
    X.append(list(map(lambda c:[c/10], lst)))
    Y.append((i+4)/10)


X=np.array(X)
Y=np.array(Y)
for i in range(len(X)):
    print(np.squeeze(X[i]),Y[i])
X.shape

model = tf.keras.Sequential([
    SimpleRNN(units = 10, return_sequences=False, input_shape=[4,1]),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
X
model.fit(X, Y, epochs=1000, verbose=0)
print(model.predict(X))

print(model.predict(np.array([[0.6],[0.7],[0.8],[0.9]])))
print(model.predict(np.array([[-0.1],[0.0],[0.1],[0.2]])))

print(model.predict(np.array([[1],[2],[3],[4]])))


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utile import to_categorical

texts = ['You are the Best Thing','Your are the Nice']

tokenizer = Tokenizer(num_words = 10, oov_token='<oov>')

tokenizer.fit_on_texts(texts)


sequences = tokenizer.texts_to_sequences(texts)

binary_results = tokenizer.sequences_to_matrix\
    (sequences, mode = 'binary')
texts
print(tokenizer.word_index)
print(sequences)
print(binary_results)


test_text = ['You are the One']
test_seq = tokenizer.texts_to_sequences(test_text)
print(test_seq)

test_bin = tokenizer.sequences_to_matrix(test_seq, mode="binary")
test_bin


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








































































