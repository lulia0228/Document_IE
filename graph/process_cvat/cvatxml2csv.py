# -*- coding: utf-8 -*-

import os
import lxml
from lxml import etree
from tqdm import tqdm
import pandas as pd

def process_cvat_xml(xml_file, output_dir):
    """
    Transforms a single XML in CVAT format to a TXT format
    XMls.
    :param xml_file: CVAT format XML
    :param image_dir: image directory of the dataset
    :param output_dir: directory of annotations with TXT format
    :return:
    """
    cvat_xml = etree.parse(xml_file)
    img_name = cvat_xml.find('filename').text
    res = []
    for obj in cvat_xml.findall('object'):
        line_data = []
        boarderNode = obj.find('polygon')
        tmp_x = []
        tmp_y = []
        for pt in boarderNode.findall('pt'):
            tmp_x.append(float(pt.find('x').text))
            tmp_y.append(float(pt.find('y').text))
        line_data.append(min(tmp_x))
        line_data.append(min(tmp_y))
        line_data.append(max(tmp_x))
        line_data.append(max(tmp_y))
        text_attr = obj.find('attributes').text
        # print(text_attr)
        text_list = text_attr.split('♥')
        if text_list[0].startswith("entity="):
            content = text_list[-1].split("text=")[-1]
            entity = text_list[0].split("entity=")[-1]
        else:
            content = text_list[0].split("text=")[-1]
            entity = text_list[-1].split("entity=")[-1]

        # print(content, '\t', entity)
        line_data.append(content)
        tag = obj.find('name').text
        line_data.append(tag)
        line_data.append(entity)
        res.append(line_data)
    new_res = sorted(res, key=lambda x:(float(x[1]),float(x[0])))
    # df = pd.DataFrame(new_res, columns=['xmin','ymin','xmax','ymax','Object','label'])
    df = pd.DataFrame(new_res, columns=['xmin','ymin','xmax','ymax','Object','label','entity'])
    df.to_csv(os.path.join(output_dir, img_name[:-4]+".csv"), index=None)

if __name__ == "__main__":
    xml_dir = "../data/xml_347_test"
    out_dir = "../data/csv_347_test_with_entity"
    count = 0
    for file in tqdm(os.listdir(xml_dir)):
        xml_file = os.path.join(xml_dir, file)
        count += 1
        process_cvat_xml(xml_file, out_dir)
    print("Deal {} jpgs.".format(count))