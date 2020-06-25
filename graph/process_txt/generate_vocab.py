# -*- coding: utf-8 -*-
# @Time    : 2020/5/4 8:51
# @Author  : Heng Li
# @File    : generate_vocab.py
# @Software: PyCharm

import pickle
import pandas as pd
import numpy as np
import re

# 路径
data_path = "../data/object_map.csv"
# 读取数据
data = pd.read_csv(data_path)
print(data.head())

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


