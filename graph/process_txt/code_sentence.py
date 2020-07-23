# -*- coding: utf-8 -*-
# @Time    : 2020/5/4 8:53
# @Author  : Heng Li
# @File    : code_sentence.py
# @Software: PyCharm

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import jieba
import sys
sys.path.append('/Users/liheng/PycharmProjects/GCN_IE/graph/process_txt')
# def seg_sentence(data, stopword_path, word_dict_path):
def seg_sentence(data):
    stopword_path = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\Graph_Extract_1\\process_txt\\stop_words.txt"
    # stopword_path = "/Users/liheng/PycharmProjects/GCN_IE/graph/process_txt/stop_words.txt"
    word_dict_path = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\Graph_Extract_1\\process_txt\\dict_all.txt"
    # word_dict_path = "/Users/liheng/PycharmProjects/GCN_IE/graph/process_txt/dict_all.txt"

    # 读取停用词
    with open(stopword_path, "r", encoding="utf-8") as f:
        stopword = f.readlines()
    stopword = [i.strip() for i in stopword]
    jieba.load_userdict(word_dict_path)

    data = data.lower()
    segment_data = jieba.cut(data)
    segment_data = [i for i in segment_data if i != ' ']
    goal_data = [i for i in segment_data if i not in stopword]
    speical_data = [i for i in segment_data if i in stopword]

    return goal_data, speical_data


# def _generate_txt_vec(data, vocab_file, stopword_path, word_dict):
def _generate_txt_vec(data):
    vocab_file = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\Graph_Extract_1\\data\\vocab.txt"
    # vocab_file = "/Users/liheng/PycharmProjects/GCN_IE/graph/data/vocab.txt"
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
    res = _generate_txt_vec("Need to email invoices Jan 2020")
    print(res)
    print(len(res))