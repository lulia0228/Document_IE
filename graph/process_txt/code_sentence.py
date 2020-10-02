# -*- coding: utf-8 -*-
# @Time    : 2020/5/4 8:53
# @Author  : Heng Li
# @File    : code_sentence.py
# @Software: PyCharm

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import jieba
import sys
import nltk
import os
import re
import pandas as pd
# sys.path.append('/Users/liheng/PycharmProjects/SROIE_GCN/graph/process_txt')
sys.path.append('D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\Item_set\\GCN_IE\\graph\\process_txt')

def match_date(str):
    pattern = "\d+/\d+/\d+|\d+-\d+-\d+"
    s = re.findall(pattern, str)
    # print("data: ",s)
    start_pos = []
    for w in s:
        start_pos.append([str.find(w),str.find(w)+len(w)])

    if start_pos != []:
        goal_word = [str[p[0]:p[1]] for p in start_pos]
        goal_word = ['('+item+')' for item in goal_word]
        new_pattern = "|".join(goal_word)
        word_split = re.split(new_pattern, str)
        word_split = [item for item in word_split if item != '' ]
        word_split = [item for item in word_split if item != None ]
        replace_split = ["ddmmyy" if item in s else item for item in word_split]
        return replace_split
    else:
        return [str]

def match_time(str):
    pattern = "\d+\:\d+\:\d+|\d+\:\d+"
    s = re.findall(pattern, str)
    # print("time : ",s)
    start_pos = []
    for w in s:
        start_pos.append([str.find(w),str.find(w)+len(w)])

    if start_pos != []:
        goal_word = [str[p[0]:p[1]] for p in start_pos]
        goal_word = ['('+item+')' for item in goal_word]
        new_pattern = "|".join(goal_word)
        word_split = re.split(new_pattern, str)
        word_split = [item for item in word_split if item != '' ]
        word_split = [item for item in word_split if item != None ]
        replace_split = ["hhmmss" if item in s else item for item in word_split]
        return replace_split
    else:
        return [str]

def match_amount(str):
    pattern = "\d+\.\d+|-\d+\.\d+|\.\d+|-\.\d+"
    s = re.findall(pattern, str)
    # print("time : ",s)
    start_pos = []
    for w in s:
        start_pos.append([str.find(w),str.find(w)+len(w)])

    if start_pos != []:
        goal_word = [str[p[0]:p[1]] for p in start_pos]
        goal_word = ['('+item+')' for item in goal_word]
        new_pattern = "|".join(goal_word)
        word_split = re.split(new_pattern, str)
        word_split = [item for item in word_split if item != '' ]
        word_split = [item for item in word_split if item != None ]
        replace_split = ["aaammm" if item in s else item for item in word_split]
        return replace_split
    else:
        return [str]

# def seg_sentence(data, stopword_path, word_dict_path):
def seg_sentence(data):
    stopword_path = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\Item_set\\GCN_IE\\graph\\process_txt\\stop_words.txt"
    # stopword_path = "/Users/liheng/PycharmProjects/SROIE_GCN/graph/process_txt/stop_words.txt"
    word_dict_path = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\Item_set\\GCN_IE\\graph\\process_txt\\dict_all.txt"
    # word_dict_path = "/Users/liheng/PycharmProjects/SROIE_GCN/graph/process_txt/dict_all.txt"

    # 读取停用词
    with open(stopword_path, "r", encoding="utf-8") as f:
        stopword = f.readlines()
    stopword = [i.strip() for i in stopword]
    jieba.load_userdict(word_dict_path)
    # print(data)
    data = data.lower()
    init_word_split = data.split(" ")
    first_word_split = []
    for w in init_word_split:
        first_word_split.extend(match_time(w))
    second_word_split = []
    for w in first_word_split:
        second_word_split.extend(match_date(w))
    third_word_split = []
    for w in second_word_split:
        third_word_split.extend(match_amount(w))
    new_str = ' '.join(third_word_split)
    segment_data = jieba.lcut(new_str)

    segment_data = [i for i in segment_data if i != ' ']
    # print("jieba: ", segment_data)
    # fenci.append(segment_data)
    # nltk_data = nltk.word_tokenize(data)
    # print("nltk: ", nltk_data)
    # fenci.append(nltk_data)
    goal_data = [i for i in segment_data if i not in stopword]
    speical_data = [i for i in segment_data if i in stopword]

    return goal_data, speical_data

def _generate_sentence_input(data):
    vocab_file = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\Item_set\\GCN_IE\\graph\\data\\vocab.txt"
    # vocab_file = "/Users/liheng/PycharmProjects/SROIE_GCN/graph/data/vocab.txt"
    vocab_dict = {}
    with open(vocab_file, 'r') as file:
        vocab_list = file.readlines()
        vocab_list = [i.strip() for i in vocab_list]
        for word in vocab_list:
            if word not in vocab_dict:
                vocab_dict[word] = len(vocab_dict)+1
    vocab_dict['_UNK_'] = 0
    data, _ = seg_sentence(data)
    idxs = [vocab_dict[w] if w in vocab_dict else 0 for w in data]
    return idxs

# def _generate_txt_vec(data, vocab_file, stopword_path, word_dict):
def _generate_txt_vec(data):
    vocab_file = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\Item_set\\GCN_IE\\graph\\data\\vocab.txt"
    # vocab_file = "/Users/liheng/PycharmProjects/SROIE_GCN/graph/data/vocab.txt"
    vocab_list = []
    with open(vocab_file, 'r') as file:
        vocab_list = file.readlines()
        vocab_list = [i.strip() for i in vocab_list]
    mlb = MultiLabelBinarizer(classes=vocab_list)
    goal_data, stop_data = seg_sentence(data)
    la = mlb.fit_transform([goal_data])
    m_y = _generate_m_y_vec(stop_data)

    return  np.hstack( (la[0],m_y) )


def _generate_m_y_vec(data):
    month_l = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    year_l = [str(i) for i in range(2018, 2030)]
    tem_list = [0,0]
    for i in data:
        if i.lower() in month_l:
            tem_list[0] = 1
            break
    for i in data:
        if i in year_l:
            tem_list[1] = 1
            break

    return np.array(tem_list)


if __name__ == "__main__":
    case_list = ["NO.12, JALAN PERMAS JAYA 10,",
                 ": 27/04/18pm7/2/2107",
                 "P50C-PSI",
                 "SUB-TOTAL",
                 "WWW.DOMINOS.COM.MY/SMILE AND",
                 "RECEIVE A 30% OFF SIDE ITEM E-COUPON",
                 "CO. NO. 419060-A",
                 "T.A.S LEISURE SDN BHD (256864-P)",
                 "GST @6%: $0.49",
                 "BAR WANG RICE@PERMAS JAYA",
                 "FACEBOOK.COM/BARWANGRICE",
                 "DD: 30/07/2017",
                 "NO. 31G&33G, JALAN SETIA INDAH X ,U13/X",
                 "MR. D.I.Y. (M) SDN BHD",
                 "(CO. REG :860671-D)",
                 "TOTAL INCL. GST@6%",
                 "NO.36G JALAN BULAN BM U5/BM,",
                 "TEL / FAX : 0163307491 / 0378317491",
                 "S/P : SALES",
                 "05-07-2017 03:17 PM",
                 "05-JAN-2017 03:17:14 PM",
                 "CO. REG. NO. : 795225-A",
                 "rm150.60",
                 "rm-150.60",
                 "s.60",
                 "-.60",
                 "rm-1.60",
                 "NO.32 & 33,JALAN SR 1/9. SEKSYEN 9.",
                 "NO.: CS-20242"]
    for s in case_list:
        seg_sentence(s)

