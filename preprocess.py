# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
csv_data=pd.read_csv('../project/covid_19.csv',encoding='utf-8')
csv_df = pd.DataFrame(csv_data)
csv_df.head()
#abondon the useless data
conserve = ['发布者关注数','发布时间','点赞数','转发数','评论数']
column_names = csv_df.columns.values.tolist()
for column_name in column_names:
    if column_name not in conserve:
        csv_df = csv_df.drop(column_name,1)
csv_df = csv_df.rename(columns = {'发布者关注数':'weibo_text','发布时间':'post_time','点赞数':'likes','转发数':'forward','评论数':'comments'})
#去除标点符号，日期归一化
from zhon.hanzi import punctuation
import re
user_name = re.compile('([\u4e00-\u9fa5]|[a-zA-Z0-9])*?:')
marks = re.compile(r"[%s]+" %punctuation)
marks_eng = re.compile('[~!@#$%^&*()_+=;:><\\-/,\ [\]]')
special_marks = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
times = re.compile('[0-9]{2}:[0-9]{2}')
csv_df['weibo_text'] = csv_df['weibo_text'].str.replace(user_name,'')
csv_df['weibo_text'] = csv_df['weibo_text'].str.replace(marks,'')
csv_df['weibo_text']= csv_df['weibo_text'].str.replace(marks_eng,'')
csv_df['weibo_text'] = csv_df['weibo_text'].str.replace(special_marks,'')
csv_df['post_time'] = csv_df['post_time'].str.replace(times,'')
#去除数值中的异常值
numbers = re.compile(r'[0-9]+?')
weibos = csv_df.values.tolist()
#print(weibos[:10])
##deal with the exceptional values
for weibo in weibos:
    if numbers.match(weibo[2])==None:
        weibo[2] = '0'
    if numbers.match(weibo[3])==None:
        weibo[3] = '0'
    if numbers.match(weibo[4])==None:
        weibo[4] = '0'
times = [weibo[1] for weibo in weibos]
post_times = list(set(times))
post_times.sort(key = times.index)
#merge the data on a daily basis
# import os
# path = './weibos'
# os.mkdir(path)
times = [weibo[1] for weibo in weibos]
post_times = list(set(times))
post_times.sort(key = times.index)
#print(post_times)
#post_times = set(post_times)
weibo_texts = []
likes = []
forwards = []
comments = []
for post_time in post_times:
    weibo_text = ''
    like = 0
    forward = 0
    comment = 0
    for weibo in weibos:
        if weibo[1]==post_time:
            weibo_text += weibo[0]
            like += int(weibo[2])
            forward += int(weibo[3])
            comment += int(weibo[4])
    weibo_texts.append(weibo_text)
    likes.append(like)
    forwards.append(forward)
    comments.append(comment)
del post_times[4]
del weibo_texts[4]
del likes[4]
del forwards[4]
del comments[4]
prepareddata = pd.DataFrame({'post_times':post_times,'weibo_texts':weibo_texts,'likes':likes,'forwards':forwards,'comments':comments})
