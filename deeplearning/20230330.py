import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text  import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 학습 결과 시각화
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'black', label='Training acc')
    plt.plot(epochs, val_acc, 'red', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'black', label='Training loss')
    plt.plot(epochs, val_loss, 'red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()



train_file = tf.keras.utils.get_file('ratings_trian.text',\
                                     origin='https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt',\
                                     extract = True)

train_file

train = pd.read_csv(train_file, sep = '\t')
train = train.iloc[:30000]
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


start = time.time()

data = train['document'].apply((lambda x : word_tokenization(x)))
print("실행시간 : ", time.time()-start)
data.head()

# data.to_csv('temp_data.txt', header=None, sep="\t")


vocab_size = 15000
oov_tok = "<OOV>"

tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size)
tokenizer.fit_on_texts(data)
print("총 단어 : ", len(tokenizer.word_index))


cnt = 0
for x in tokenizer.word_counts.values():
    if x >= 5 :
        cnt += 1
print("5회이상 사용된단어 : ", cnt)



train_size=len(data)*0.7
train_size=int(train_size)
train_data = data[:train_size]
valid_data = data[train_size:]
len(train_data)
len(valid_data)

train_y = train['label'][:train_size]
valid_y = train['label'][train_size:]
train_y[:10]
valid_y[:10]
train_data[:10]

train_data = tokenizer.texts_to_sequences(train_data)
valid_data = tokenizer.texts_to_sequences(valid_data)
train_data[:10]
valid_data[:10]


max(len(x) for x in valid_data)
max_len = max(len(x) for x in train_data)
print('문장 최대길이', max_len)


train_pad = pad_sequences(train_data,padding="post", maxlen=max_len)
valid_pad = pad_sequences(valid_data,padding="post", maxlen=max_len)
len(train_pad[0])
len(train_pad[1])
train_pad[0]


# 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

model = Sequential([
    Embedding(vocab_size , 32),
    Bidirectional(LSTM(32, return_sequences=True)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
    ])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()



history = model.fit(train_pad, train_y,
                    validation_data = (valid_pad, valid_y),
                    batch_size=64, epochs=10, verbose=1)



plot_history(history)

def preprocessing(df):
    df['document'] = df['document'].str.replace("[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ ]","")
    df = df.dropna()
    test_label = df['label']
    test_data = df['document'].apply((lambda x : word_tokenization(x)))
    
    test_data = tokenizer.texts_to_sequences(test_data)
    test_data = pad_sequences(test_data, padding="post", maxlen=max_len)
    return test_data, test_label




# 검증
test=train.iloc[-1000:]
test.info()


test_data, test_label = preprocessing(test)
len(test_data)
test_data[0]
test_label[:10]
len(test_data[0])
len(test_label)
test_data[0]



model.evaluate(test_data, test_label)


mydata=train.iloc[500:510]
mydata.info()
len(test["label"])
mydata["label"][2:3]



# mydata = pd.DataFrame({'id': [1, 2],
#                        'document': ["영화 재밌어요", '영화 재미없어요'],
#                        'label': [1, 0]
#                       })
mydata.info()

X,Y = preprocessing(mydata)
len(X)
model.evaluate(X,Y)
pred = model.predict(X)
len(pred)


sigmoid = lambda x : 1 if x > 0.5 else 0

for i in range(0,len(mydata['label'])) :
    print(sigmoid(np.mean(pred[i])), mydata["label"][i:i+1])
















