# -*- coding: utf-8 -*-
# @Time    : 2020/5/4 8:51
# @Author  : Heng Li
# @File    : generate_vocab.py
# @Software: PyCharm

import pickle
import pandas as pd
import numpy as np
import re
import os

# 路径
# data_path = "../data/object_map.csv"
csv_dir = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\Item_set\\GCN_IE\\graph\\data\\csv_data"
# 读取数据
data = pd.concat([pd.read_csv(os.path.join(csv_dir,file )) for file in os.listdir(csv_dir)])
# print(len(data), data.head(),type(data))
# print(data.head())

## 分词、建立词到id的映射表
import jieba
stopword_path = "stop_words.txt"
word_dict_path = "dict_all.txt"

# 读取停用词
with open(stopword_path,"r",encoding="utf-8") as f:
    stopword = f.readlines()
stopword = [i.strip() for i in stopword]

jieba.load_userdict(word_dict_path)

def seg_sentence(data):
    segment_data = jieba.cut(data)
    segment_data = [i for i in segment_data if i != ' ']
    goal_data = [i for i in segment_data if i not in stopword]
    speical_data = [i for i in segment_data if i in stopword]

    return goal_data, speical_data

text_list = data['Object'].to_list()
vocab_list = []
for i in text_list:
        goal,_  =  seg_sentence(i)
        vocab_list.extend(goal)

vocab_list = [i.lower() for i in vocab_list]
vocab_list = list(set(vocab_list))

with open("../data/vocab.txt", 'w') as f:
    for i in vocab_list:
        f.writelines(i+'\n')


