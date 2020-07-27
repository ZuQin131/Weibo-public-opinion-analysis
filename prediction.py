# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:44:39 2020

@author: CielYoung
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import matplotlib.pyplot as plt
from keras.layers import Conv1D,Flatten
from keras import layers
from keras import backend as K
import keras

X=[]
for i in range(22):
    X.append([i])

def draw_residuals(y_true,y_pred):
    plt.title(u'heat prediction')
    plt.xlabel(u'date')
    plt.ylabel(u'heat')
    plt.axis([0,25,0,25])
    plt.grid(True)
    plt.plot(X,y_true,'k.')
    plt.plot(X,y_pred,'bo')
    plt.show()

weibo_excel = pd.read_excel('weibo.xlsx',encoding='utf-8')
weibos_df = pd.DataFrame(weibo_excel)
#weibo_texts = weibos['weibo_texts']
weibo_texts = weibos_df['weibo_texts'].values.tolist()
#分词，去停用词
import jieba
import re
words = []
for weibo in weibo_texts:
    words.append(jieba.lcut(weibo))
with open("../project/stopwords.txt",'r',encoding='utf-8') as f:
    stopwords = f.readlines()
    stopwords = [stopword.split('\n')[0]for stopword in stopwords]
seq = ['回复','转发微博','转发','微博','原图','超话','组图','视频','地图']
stopwords.extend(seq)
#print(stopwords[500:550])
eng = re.compile(r'[A-Za-z0-9]+?')
texts = []
for word in words:
    word = [w for w in word if w not in stopwords]
    word = [w for w in word if eng.match(w)==None]
    texts.append(word)
#tfidf文本预处理
weibos = []
for i in range(len(texts)):
    weibos.append('')
for i in range(len(texts)):
    for word in texts[i]:
        weibos[i] += word + ' '
#weibos[1]
#tfidf文本表示法

vec=CountVectorizer(min_df=3,lowercase = False,ngram_range=(1,1))
X=vec.fit_transform(weibos)
trans=TfidfTransformer()
tfidf=trans.fit_transform(X)
print (len(vec.get_feature_names()))


# train word2vec skipgram model
from gensim.models import Word2Vec
import time
# set model parameters
num_features = 100    # Word vector dimensionality
min_word_count = 10    # Minimum word count
context = 5           # Context window size
sg = 1                # skipgram=1, cbow=0
num_workers = 4       # Number of threads to run in parallel

# train the model
%time model_sg100 = Word2Vec(texts, size=num_features, window=context, min_count=min_word_count, sg=sg, workers=num_workers)
word_vectors = model_sg100.wv.vectors
vocab = model_sg100.wv.vocab.keys()
vocab = list(vocab)
feature_names = vec.get_feature_names()


#找到文本中每个单词对应的tfidf值
#用tfidf值乘以word vector求得文本向量
def get_text_vec(text):
    vec_text = np.zeros(100)
    for word in text:
        if word in feature_names:
            tfidf_index = feature_names.index(word)
            if word in vocab:
                vec_text +=model_sg100.wv[word]*tfidf[texts.index(text),tfidf_index]
        else:
            continue
    return vec_text
#print(vec_text)
#获得所有文本的加权向量
vec_texts = np.array([])
for text in texts:
    vec_text = get_text_vec(text)
    vec_texts = np.hstack((vec_texts,vec_text))
vec_texts = vec_texts.reshape(102,100)
import math
forwards = weibos_df['forwards'].values.tolist()
likes = weibos_df['likes'].values.tolist()
comments = weibos_df['comments'].values.tolist()

heats = []
for i in range(len(texts)):
    summation = int(forwards[i])+int(likes[i])+int(comments[i])+1
    rev_sum = 1/summation
    heats.append((-1) * math.log(rev_sum))


vec_train = np.array(vec_texts[:int(vec_texts.shape[1] * 0.8)])
vec_test = np.array(vec_texts[int(vec_texts.shape[1] * 0.8):])
heat_train = np.array(heats[:int(vec_texts.shape[1] * 0.8)])
heat_test = np.array(heats[int(vec_texts.shape[1] * 0.8):])

#DNN
model1 = Sequential() #建立模型
model1.add(Dense(input_dim = 100, output_dim = 64)) #添加输入层、隐藏层的连接
model1.add(Activation('sigmoid')) #以Relu函数为激活函数
model1.add(Dense(input_dim = 64, output_dim = 64)) #添加隐藏层、隐藏层的连接
model1.add(Activation('sigmoid')) #以Relu函数为激活函数
model1.add(Dense(input_dim = 64, output_dim = 64)) #添加隐藏层、隐藏层的连接
model1.add(Activation('sigmoid')) #以Relu函数为激活函数
model1.add(Dense(input_dim = 64, output_dim = 1)) #添加隐藏层、输出层的连接
model1.compile(loss='mean_squared_error', optimizer='sgd',metrics = [tf.keras.metrics.MeanAbsoluteError()])
model1.fit(vec_train, heat_train,validation_split=0.2, batch_size=32, epochs=32)

test_scores = model1.evaluate(vec_test, heat_test, verbose=0)
print(test_scores)
prediction1 = model1.predict(vec_test)

#按时间顺序排列文章
texts_seq = []
for i in range(len(vec_texts)):
    texts_seq.append(vec_texts[101-i])
texts_seq = np.asarray(texts_seq)
texts_seq = texts_seq.reshape(102,100,1)
heats_seq = []
for i in range(len(heats)):
    heats_seq.append(heats[101-i])
heats_seq = np.asarray(heats_seq)
vecq_train = np.array(texts_seq[:int(texts_seq.shape[1] * 0.8)])
vecq_test = np.array(texts_seq[int(texts_seq.shape[1] * 0.8):])
heatq_train = np.array(heats_seq[:int(texts_seq.shape[1] * 0.8)])
heatq_test = np.array(heats_seq[int(texts_seq.shape[1] * 0.8):])

#RNN


#def r_square(y_true, y_pred):
#    SSR = K.mean(K.square(y_pred-K.mean(y_true)),axis=-1)
#   SST = K.mean(K.square(y_true-K.mean(y_true)),axis=-1)
#   return SSR/SST
print('Build model...')
model2 = Sequential()

# Add a LSTM layer with 128 internal units.
model2.add(layers.LSTM(128,input_shape=(100,1)))

model2.add(layers.Dense(1))
model2.compile(loss='mean_squared_error',optimizer='sgd',metrics=[tf.keras.metrics.MeanAbsoluteError()])
print('Train...')
model2.fit(vecq_train, heatq_train, batch_size=32, epochs=32, validation_data=(vecq_test, heatq_test))
score = model2.evaluate(vecq_test, heatq_test,batch_size=32)
print('Test score:', score)
prediction2 = model2.predict(vecq_test)

model21 = Sequential()

# Add a LSTM layer with 128 internal units.
model21.add(layers.LSTM(128,input_shape=(100,1)))

model21.add(layers.Dense(1))
model21.compile(loss='mean_squared_error',optimizer='sgd',metrics=[tf.keras.metrics.MeanAbsoluteError()])
model21.fit(vecq_train, heatq_train,validation_split=0.2, batch_size=32, epochs=32)
prediction21 = model21.predict(vecq_test)

#CNN
model3 = Sequential()
model3.add(Conv1D(32,2,activation = 'relu',input_shape = (100,1)))
model3.add(Flatten())
model3.add(Dense(64,activation = 'relu'))
model3.add(Dense(1))
model3.compile(loss = 'mse',optimizer='adam',metrics = [tf.keras.metrics.MeanAbsoluteError()])
# inputs = layers.Input(shape = (100,1))
# x_train = Conv1D(32,4,activation = 'relu',input_shape = (100,1))(vecq_train)
vec_train=vec_train.reshape(80,100,1)
vec_test=vec_test.reshape(22,100,1)
model3.fit(vec_train, heat_train,validation_split=0.2, batch_size=32, epochs=32)
score = model3.evaluate(vec_test, heat_test)
print('Test score:', score)
prediction3 = model3.predict(vec_test)
draw_residuals(heat_test,prediction1)
draw_residuals(heatq_test,prediction2)
draw_residuals(heatq_test,prediction21)
draw_residuals(heat_test,prediction3)