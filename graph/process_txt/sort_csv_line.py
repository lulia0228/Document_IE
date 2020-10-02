# -*- coding: utf-8 -*-
# @Time    : 2020/8/22 19:35
# @Author  : Heng Li
# @File    : sort_csv_line.py
# @Software: PyCharm

import pandas as pd
import os

csv_orgin = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\Item_set\\SROIE_GCN\\graph\\data\\csv_sroie"
csv_save = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\Item_set\\SROIE_GCN\\graph\\data\\csv_sroie_new"

for file in os.listdir(csv_orgin):
    df = pd.read_csv(os.path.join(csv_orgin, file))
    df = df.sort_values(by=['ymin', 'xmin'])
    df.to_csv(os.path.join(csv_save, file), index=None)
