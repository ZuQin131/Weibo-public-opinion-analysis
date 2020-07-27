# -*- coding: utf-8 -*-
'''
基于字典的情感分析
'''

import jieba
import csv
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_pro(emotions):
    # 读取文件
    weibo = pd.read_csv("./topic/#新冠肺炎#0501-0120.csv", encoding='utf-8')

    contents = weibo["微博正文"].tolist()
    likes = weibo['点赞数'].tolist()

    def get_source(type):
        # 返回情感类型的词语及强度
        dictionary = pd.read_csv("./topic/情感词汇本体.csv", encoding='utf-8')
        emotion = dictionary.loc[dictionary['情感分类'] == type][['词语', '强度']]
        words = emotion['词语'].tolist()
        intensity = emotion['强度'].tolist()
        return words, intensity

    def sum_e():
        # 将同一天的情绪值加总
        evalue = pd.read_csv('./topic/#新冠肺炎#0501-0120.csv', encoding='utf-8')
        evalue = evalue[['发布时间', 'PA', 'PE', 'PD', 'PH', 'PG', 'PB', 'PK', 'NA', 'NB', 'NJ', 'NH', 'PF', 'NI', 'NC',
                         'NG', 'NE', 'ND', 'NN', 'NK', 'NL', 'PC']]
        evalue['发布时间'] = pd.to_datetime(evalue['发布时间'])
        evalue = evalue.groupby("发布时间").sum()
        evalue.to_csv('./topic/emotionValue.csv', mode='w', header=True, encoding='utf-8', index=True)
        print("完成情绪值汇总")

    for emotion in emotions:
        words, intensity = get_source(emotion)
        values = []

        for content, like in zip(contents, likes):
            content_list = list(jieba.cut(content, cut_all=False, HMM=False))
            count = 0
            # i = 0

            # 计算每条微博的情绪值 = 情绪词*强度*(点赞数+1)
            for word in content_list:
                if word in words:
                    count += 1
                # i += 1

            values.append(count * (int(like) + 1))
        # 写入一种情绪值
        weibo[emotion] = pd.DataFrame(values)
        weibo.to_csv('./topic/#新冠肺炎#0501-0120.csv', mode='w', header=True, encoding='utf-8', index=False)
        print(emotion + "已编入")

        sum_e()


def visualize(Es):
    # 可视化情绪变化
    evalues = pd.read_csv('./topic/emotionValue.csv', encoding='utf-8')
    evalues.set_index('发布时间', inplace=True)
    evalues = evalues[Es]
    evalues.plot()
    plt.title('新冠肺炎')
    plt.show()

def visualize_lda(Es):
    # 可视化情绪变化
    evalues = pd.read_csv('./topic/emotionValue.csv', encoding='utf-8')
    evalues.set_index('topic', inplace=True)
    evalues = evalues[Es]
    evalues.plot()
    plt.title('新冠肺炎')
    plt.show()




if __name__ == '__main__':
    # 基于字典的词频分析，将日期相同的微博正文放在一起，用字典统计每天的文本，输出一个向量 ：21个分量
    emotions = ['PA', 'PE', 'PD', 'PH', 'PG', 'PB', 'PK', 'NA', 'NB', 'NJ', 'NH', 'PF', 'NI', 'NC', 'NG', 'NE', 'ND',
              'NN', 'NK', 'NL', 'PC']

    # 统计每条微博的情绪值
    # read_pro(emotions)

    # 可视化
    # 悲伤、失望、贬责
    Es = ['NB', 'NJ', 'NN']
    # visualize(Es)
    visualize_lda(Es)
    # # 慌张、羞愧、怀疑
    Ns = ['NI', 'NG', 'NL']
    #visualize(Ns)
    visualize_lda(Ns)

