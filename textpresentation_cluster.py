# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:39:04 2020

@author: CielYoung
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud
import random
#K—means聚类提取话题
#产生初始中心点
def create_particle(k,dataset):
    dataset = dataset.tolist()
    particle = random.sample(dataset,k)
    return particle
#计算欧式距离
def euc_distance(dot1,dot2):
    difference = np.asarray(dot1) - np.asarray(dot2)
    return np.linalg.norm(difference)
#计算点与聚类中心的距离
def dotsToParticles(dots, particles):
    dots_distance_list = []
    for dot in dots:
        dot_to_particle = []
        for particle in particles:
            dot_to_particle.append(euc_distance(dot,particle))
        dots_distance_list.append(dot_to_particle)
    return dots_distance_list
#新的聚类中心点
def center_of_category(kind,length):
    summation = np.zeros(length)
    for i in kind:
        summation += vec_texts[i]   
    particle = summation/len(kind)
    return particle.tolist()
#聚类，更新聚类中心
def classification(dots,particles,length,kinds,points):
    dots_distance_list = dotsToParticles(dots,particles)
    min_distance_dot = dots_distance_list[0][0]
    index_x = 0
    index_y = 0
    for i in range(len(dots)):
        if min(dots_distance_list[i])< min_distance_dot:
            min_distance_dot = min(dots_distance_list[i])
            #更新最小距离的索引
            index_x = i
            index_y = dots_distance_list[i].index(min_distance_dot)
    kinds[index_y].append(points.index(dots[index_x]))
    dots.remove(dots[index_x])
    newparticles = []
    for kind in kinds:
        newparticles.append(center_of_category(kind,length))
    particles = newparticles
def generate_word_cloud(kind,name):
    vocabulary = []
    remove_words = ['新冠','肺炎','新型','疫情','目前','确诊','患者','病例']
    for i in kind:
        vocabulary.extend(texts[i])
    #print(type(vocabulary))
    #for word in remove_words:
    vocabulary = [word for word in vocabulary if word not in remove_words]
    wordlist = ' '.join(vocabulary)
    w = WordCloud(
        background_color = 'white',
        max_words = 400,
        #设置字体
        font_path = "C:\Windows\Fonts\SIMYOU.ttf",
        height = 2000,
        width = 2000,
        max_font_size = 400,
        random_state = 30,
        collocations=False
    )
    myword = w.generate(wordlist)
    plt.imshow(myword)
    plt.axis('off')
    plt.show()
    w.to_file('%s.png'%name)

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
print ("number of features: "+str(len(vec.get_feature_names())))
print (tfidf.toarray())

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
#print(vocab)
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
#文本向量聚类
text_dim = 100
text_category = 8
particles = create_particle(text_category,vec_texts)
vec_tem = vec_texts.tolist()
points = vec_texts.tolist()
kinds_text = []
#particles = centers.tolist()
#len(vec_tem)
for particle in particles:
    kinds_text.append([vec_tem.index(particle)])
for particle in particles:
    vec_tem.remove(particle)
# classification(vec_tem,particles)
# print(len(vec_tem))
# for i in vec_tem:
#         print(points.index(i),end = ' ')
while len(vec_tem)>0:
    print(kinds_text)
#     for i in vec_tem:
#         print(points.index(i),end = ' ')
    classification(vec_tem,particles,text_dim,kinds_text,points)
#     for i in vec_tem:
#         print(points.index(i))
#     print(kinds)
print("The cluster result:")
for kind in kinds_text:
    print(kind)
if __name__ == '__main__':
    for kind in kinds_text:
        name = 'text_category'+ str(kinds_text.index(kind)+1)
        generate_word_cloud(kind,name)
#基于tfidf的聚类
word_dim = 100
text_category = 8
all_texts = tfidf.toarray().tolist()
text_tem = tfidf.toarray().tolist()
particles = create_particle(text_category,tfidf.toarray())
kinds_tfidf = []
for particle in particles:
    kinds_tfidf.append([all_texts.index(particle)])
for particle in particles:
    text_tem.remove(particle)
while len(text_tem)>0:
    print(kinds_tfidf)
    classification(text_tem,particles,word_dim,kinds_tfidf,all_texts)
print("The cluster result:")
for kind in kinds_tfidf:
    print(kind)
for kind in kinds_tfidf:
        name = 'tfidf_category'+ str(kinds_tfidf.index(kind)+1)
        generate_word_cloud(kind,name)