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
import code_sentence

# 路径
# data_path = "../data/object_map.csv"
csv_dir = "/Users/liheng/PycharmProjects/SROIE_GCN/graph/data/csv_sroie_new"
# 读取数据
data = pd.concat([pd.read_csv(os.path.join(csv_dir,file)) for file in os.listdir(csv_dir)], keys=os.listdir(csv_dir))
# print(len(data), data.head(),type(data))
print(data.head())
# dd = data.loc[data['label'] == 'o']
# print(dd)
# exit()

## 分词、建立词到id的映射表
import jieba
stopword_path = "stop_words.txt"
word_dict_path = "dict_all.txt"

# 读取停用词
with open(stopword_path,"r",encoding="utf-8") as f:
    stopword = f.readlines()
stopword = [i.strip() for i in stopword]

jieba.load_userdict(word_dict_path)

# def seg_sentence(data):
#     segment_data = jieba.lcut(data)
#     segment_data = [i for i in segment_data if i != ' ']
#     goal_data = [i for i in segment_data if i not in stopword]
#     speical_data = [i for i in segment_data if i in stopword]
#
#     return goal_data, speical_data

text_list = data['Object'].to_list()
label_list = data['label'].to_list()

key_vocab = {'O':[],'COMPANY':[],'ADDRESS':[],'DATE':[],'TOTAL':[]}

for i in range(len(text_list)):
    print("origin : ", text_list[i])
    goal,_  =  code_sentence.seg_sentence(text_list[i])
    try:
        key_vocab[label_list[i]].extend(goal)
    except Exception as error:
        print(error)

def get_key_vocab(key_vocab_list):
    tmp_dict = {}
    for word in key_vocab_list:
        if word not in tmp_dict:
            tmp_dict[word] = 1
        else:
            tmp_dict[word] += 1
    tmp_list = []
    for k, v in tmp_dict.items():
        if v > 3:
            # print(k, v)
            tmp_list.append(k)
    return tmp_list

vocab_list = []
for k,v in key_vocab.items():
    vocab_list.extend(get_key_vocab(v))

vocab_list = [i.lower() for i in vocab_list]
vocab_list = list(set(vocab_list))

import re
# pattern = r'(\d+).(\d+)'
# print(re.match(pattern,"sd14.001"))
# exit()
def filter(vocab_list):
    res = []
    pattern = r'(\d+).(\d+)'
    for w in vocab_list:
        if re.fullmatch(pattern, w):
            print("&&&", w)
            continue

        if w.isnumeric():
            print("***", w)
            continue

        res.append(w)
    return res


final_vocab_list = filter(vocab_list)

print(len(final_vocab_list))
for item in final_vocab_list:
    print("final vocab :", item)

with open("../data/vocab.txt", 'w') as f:
    for i in final_vocab_list:
        f.writelines(i+'\n')


