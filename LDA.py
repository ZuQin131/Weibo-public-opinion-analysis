# -*- coding: utf-8 -*-
'''
主题模型
'''

import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg
import re
from gensim import corpora, models


def text_pro(text, rule_words):
    '''
    文本预处理函数，将输入的文本句子根据词性标注做分词，判断是否保留词
    :param text: 文本句子，字符串型
    :return: 符合规则的分词结果
    '''

    words = pseg.cut(text)  # 分词
    seg_list = []  # 列表用于存储每个文件的分词结果
    for word in words:  # 循环得到每个分词
        if word.flag in rule_words and is_fine_word(word.word):
            seg_list.append(word.word)  # 将分词追加到列表
    return seg_list


def is_fine_word(word):
    '''
    文本预处理：词是否为满足词长，不为停用词，为中文
    :return: 真，假
    '''
    min_length = 2  # 最短词长

    rule = re.compile(r"^[\u4e00-\u9fa5]+$")
    if len(word) >= min_length and word not in stopwords and re.search(rule, word):
        return True
    else:
        return False


def text_pro_set():
    '''
    文本预处理，运行前设置
    :return: 文本，list of list
    '''

    # 读入数据
    weibo_excel = pd.read_excel('./topic/weibo.xlsx', encoding='utf-8')
    weibos = pd.DataFrame(weibo_excel)
    weibo_texts = weibos['weibo_texts'].values.tolist()

    # 停用词设置
    with open("./topic/stopwords.txt", 'r', encoding='utf-8') as f:
        global stopwords
        stopwords = f.readlines()
        stopwords = [stopword.split('\n')[0] for stopword in stopwords]
        stopwords.extend(
            ['\n', '\t', ' ', '回复', '转发微博', '转发', '微博', '原图', '超话', '组图', '视频', '地图', '显示', '新闻', '没有', '截至', '表示','情况','进行','报道'])

    # 词性设置：只保留状态词、名动词、动词、时间词、其他名词、人名、地名、名词、习用语、简称略语、成语、形容词、名形词
    # rule_words = ['z', 'vn', 'v', 't', 'nz', 'nr', 'ns', 'n', 'l', 'i', 'j', 'an', 'a']
    rule_words = ['vn', 'v', 'nz', 'nr', 'ns', 'n', 'l', 'i', 'j', 'an', 'a']

    # 运行文本处理
    texts = []
    for text in weibo_texts:
        text_list = text_pro(text, rule_words)
        texts.append(text_list)

    return texts


def print_topic(texts, n_topics, n_words):
    '''
    输出主题模型结果
    :param n_topics: LDA主题数量
    :param n_words: 主题的词范围（数量）
    :return: 输出公共主题，单日主题
    '''

    # 准备语料库/Vocab
    vocabs = corpora.Dictionary(texts)  # generate vocabulary dictionary
    corpus = [vocabs.doc2bow(word) for word in texts]  # generate corpus for training model
    corpora.MmCorpus.serialize('weibo.mm', corpus)

    # 训练LDA模型
    # 设定主题个数：
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=vocabs, num_topics=n_topics, random_state=1)
    # 展示训练结果
    # lda.print_topics(num_topics=n_topics, num_words=n_words)

    corpus_lda = lda[corpus]
    main_topics = []
    for i in range(len(texts)):
        # print(corpus_lda[i])
        # 选取单日"权重最大"的主题
        main_topics.append(sorted(corpus_lda[i], key=lambda x: x[1], reverse=True)[0])

    # 梳理主题结果：排除在每个主题列表中都出现的词；公共主题，单日主题
    # 整理单日的主题
    for t in range(0, n_topics):
        name = 't' + str(t)
        locals()['t' + str(t)] = []
        for i in range(0, n_words):
            locals()['t' + str(t)].append((lda.show_topic(t, n_words)[i])[0])

    # 提取公共主题
    p_topic = locals()['t' + str(0)]
    for t in range(1, n_topics):
        p_topic = list(set(p_topic).intersection(set(locals()['t' + str(t)])))
    print("public topic:", p_topic)

    # 提取单日"独有的"主题
    day_topics = []
    for t in range(0, n_topics):
        locals()["t" + str(t)] = list(set(locals()["t" + str(t)]).difference(set(p_topic)))
        day_topics.append(locals()["t" + str(t)])
        print(t, ":", locals()["t" + str(t)])

    # 返回变量：  每一天权重最高的topic的数据
    days_topic = []
    for i in range(len(texts)):
        main_topic = main_topics[i]
        days_topic.append(day_topics[main_topic[0]])

    return main_topics, p_topic, days_topic


if __name__ == '__main__':
    # 文本预处理
    texts = text_pro_set()

    # 训练、使用LDA
    main_ts, public_t, days_t = print_topic(texts, 5, 35)
    # 保存主题编号
    # main_t = [str(i[0]) for i in main_ts]
    # main_t.reverse()

    #保存文字
    days_t = [str(i) for i in days_t]
    days_t.reverse()

    # 保存数据
    df = pd.read_csv("./topic/emotionValue.csv", encoding='utf-8')

    df['topic'] = pd.DataFrame(days_t)
    df.to_csv('./topic/emotionValue.csv', mode='w', header=True, encoding='utf-8', index=False)

    print(public_t)
    print(df.head())

