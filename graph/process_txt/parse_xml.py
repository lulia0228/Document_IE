#--coding:utf-8--
'''
@Time   : 2020/7/24
@Author : Heng Li
@Email  : liheng@elensdata.com
'''
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
        content = obj.find('attributes').text.split('=')[-1]
        line_data.append(content)
        tag = obj.find('name').text
        line_data.append(tag)
        res.append(line_data)
    new_res = sorted(res, key=lambda x:(float(x[1]),float(x[0])))
    df = pd.DataFrame(new_res, columns=['xmin','ymin','xmax','ymax','Object','label'])
    df.to_csv(os.path.join(output_dir, img_name[:-4]+".csv"),index=None)


if __name__ == "__main__":
    xml_dir = "../data/cvat_xmls"
    img_dir = "../data/cvat_pngs"
    csv_dir = "/Users/liheng/PycharmProjects/GCN_IE/graph/data/csv_data"
    count = 0
    for file in tqdm(os.listdir(xml_dir)):
        xml_file = os.path.join(xml_dir, file)
        img_file = os.path.join(img_dir, file[:-4]+".png")
        if not os.path.exists(img_file):
            continue
        count += 1
        process_cvat_xml(xml_file, csv_dir)
    print("Deal {} pngs.".format(count))