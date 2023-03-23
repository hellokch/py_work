# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 09:22:45 2023

@author: KITCOOP
"""

##################################################
# Fashion-MNIST 데이터셋 다운로드
import matplotlib.pyplot as plt
import pandas as pd
pd.__version__
import numpy as np
np.__version__
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


#1. data 수집
from tensorflow.keras.datasets.fashion_mnist import load_data
(ox_train, oy_train), (ox_test, oy_test) = load_data()
print(ox_train.shape,ox_test.shape) #(60000, 28, 28) (10000, 28, 28)
class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
oy_train[:10] #[9, 0, 0, 3, 0, 2, 7, 2, 5, 5]
ox_train[0]
plt.imshow(ox_train[8])





#2. 데이터 전처리
#이미지 데이터 정규화
x_train = ox_train/255 #minmax 정규화 (x-min)/(max-min)
x_test = ox_test/255
#레이블을 onehot인코딩하기
y_train = to_categorical(oy_train)
y_test = to_categorical(oy_test)







# ox_train---전처리 ----> x_train

#검증데이터 분리. (훈련:검증)=(7:3)
x_train,x_val,y_train,y_val = \
  train_test_split(x_train,y_train,test_size=0.3,random_state=777)
  
model = Sequential()  #모델 생성
model.add(Flatten(input_shape = (28,28)))
model.add(Dense(64,activation="relu",input_shape=(784,)))
model.add(Dense(32,activation="relu"))
model.add(Dense(10,activation="softmax"))
model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['acc'])




history=model.fit(x_train,y_train,epochs=30,batch_size=128,validation_data=(x_val,y_val)) 

#4. 검증

history.history["loss"][29] #훈련데이터 손실함수값
history.history["acc"][29] #훈련데이터 정확도
history.history["val_loss"][29] #검증데이터 손실함수값
history.history["val_acc"][29] #검증데이터 정확도값

model.evaluate(x_test, y_test)
results = model.predict(x_test)
np.argmax(results[:10],axis=-1)
np.argmax(y_test[:10],axis=-1)


history.history["loss"]




from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(np.argmax(y_test,axis=-1),\
                    np.argmax(results,axis=-1))


#heatmap으로 출력하기
import matplotlib. pyplot as plt
import seaborn as sns
plt.figure(figsize=(7,7))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('predicted label',fontsize=15)
plt.ylabel('true label',fontsize=15)

#history1 데이터로 loss,acc -훈련데이터,검증데이터부분을 선 그래프로 출력하기

his_dict = historyl.history
loss = his_dict['loss']
val_loss = his_dict ['val_loss']
epochs
range(1, len(loss) + 1)
fig = plt. figure(figsize = (10, 5))
ax1 = fig.add_subplot (1, 2, 1)
ax1.plot(epochs, loss, color = 'blue', label = "train loss")
ax1.plot(epochs, val_loss, color = "orange", label = "val_loss")
axl.set_title('train and val loss")
axl.set_xlabel('epochs')
axl.set_ylabel( 'loss")
ax1.legend()
#정확도 출력 그래프
acc= his_dict['acc']
val_acc = his_dict['vallacc"]
ax2 = fig. add_subplot (1, 2, 2)
ax2.plot(epochs, acc, color = "blue', label = 'train_ acc')
ax2.plot(epochs, val_acc, colon = "orange', label = 'val_acc')
ax2.set title( 'train and val acc')
ax2.set_xlabel('epochs')
ax2.set_ylabel('acc")
ax2.legend()
plt.show()

plt.xticks(range(10),class_names, rotation=45)
plt.yticks(range(10),class_names, rotation=0)

plt.show()



#history1 데이터로 loss,acc -훈련데이터,검증데이터부분을 선 그래프로 출력하기
import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

# 손실함수 그래프 출력
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(epochs, loss, 'bo', label='Training loss')
ax1.plot(epochs, val_loss, 'b', label='Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# 정확도 그래프 출력
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

ax2.plot(epochs, acc, 'ro', label='Training accuracy')
ax2.plot(epochs, val_acc, 'r', label='Validation accuracy')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

# 클래스별 정확도 그래프 출력
class_acc = history_dict['class_accuracy']
val_class_acc = history_dict['val_class_accuracy']

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))

ax3.plot(epochs, class_acc, 'go', label='Training class accuracy')
ax3.plot(epochs, val_class_acc, 'g', label='Validation class accuracy')
ax3.set_title('Training and validation class accuracy')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Class accuracy')
ax3.legend()

plt.show()





model2 = Sequential()  #모델 생성
model2.add(Flatten(input_shape = (28,28)))
model2.add(Dense(256,activation="relu"))
model2.add(Dense(128,activation="relu"))
model2.add(Dense(64,activation="relu"))
model2.add(Dense(32,activation="relu"))
model2.add(Dense(10,activation="softmax"))
model2.summary()


model2.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['acc'])



history2=model2.fit(x_train,y_train,epochs=30,batch_size=128,validation_data=(x_val,y_val)) 

model.evaluate(x_test,y_test)
model2.evaluate(x_test,y_test)

#모델 저장하기
model.save("fashion1.h5")
model2.save("fashion2.h5")

from keras.models import load_model
m1 = load_model("fashion1.h5")
m2 = load_model("fashion2.h5")
m1.evaluate(x_test,y_test)
m2.evaluate(x_test,y_test)


plt.imshow(x_test[6])

count = 0
for idx in range(len(results)):
    number_sol = np.argmax(results,axis=1)[idx]
    number_y = np.argmax(y_test,axis=1)[idx]
    if number_y != number_sol:
        plt.subplot(4,4,count+1)
        plt.axis('off')
        plt.imshow(x_test[idx].reshape(28,28))
        plt.title('Pred:%d,\n lab:%d' %
                  (number_sol,number_y),fontsize=15)
        count += 1
        if count > 15 : break

plt.tight_layout()
plt.show()


from PIL import Image
im = Image.fromarray(ox_test[0])
im.save("img/fashion.jpg")

readimage = Image.open('img/fashion.jpg')
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




#############################
#  이항분류 : 분류의 종류가 2종류인 경우
import pandas as pd
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
test_x,test_y = wine_np[train_idxL,L-1],wine_np[train_idx:,-1]
test_x.shape
test_y.shape