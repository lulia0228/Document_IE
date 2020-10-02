#--coding:utf-8--

import os
import json
import requests
import base64
import glob
import xml.dom.minidom as XDM
import numpy as np
from lxml import etree
import pandas as pd
import cv2
import shutil

def save_xml(res_json,reserve_xmlpath):
    # xml_path = img_filename.replace(img_filename.split(".")[-1], "xml")

    # create a empty xml doc
    doc = XDM.Document()
    # create a root node
    root = doc.createElement("annotation")
    doc.appendChild(root)

    filenameNode = doc.createElement("filename")
    filenameNode.appendChild(doc.createTextNode(res_json["image_name"]))
    folderNode = doc.createElement("folder")
    root.appendChild(filenameNode)
    root.appendChild(folderNode)

    sourceNode = doc.createElement("source")
    sourceImageNode = doc.createElement("sourceImage")
    sourceAnnotationNode = doc.createElement("sourceAnnotation")
    sourceAnnotationNode.appendChild(doc.createTextNode("CVAT"))
    sourceNode.appendChild(sourceImageNode)
    sourceNode.appendChild(sourceAnnotationNode)
    root.appendChild(sourceNode)

    imagesizeNode = doc.createElement("imagesize")
    nrowNode = doc.createElement("nrows")
    nrowNode.appendChild(doc.createTextNode(res_json["imagesize"][0]))
    ncolsNode = doc.createElement("ncols")
    ncolsNode.appendChild(doc.createTextNode(res_json["imagesize"][1]))
    imagesizeNode.appendChild(nrowNode)
    imagesizeNode.appendChild(ncolsNode)
    root.appendChild(imagesizeNode)

    # create filename
    for i, text in enumerate(res_json["text"]):
        objectNode = doc.createElement("object")

        nameNode = doc.createElement("name")
        # nameNode.appendChild(doc.createTextNode("box"))
        nameNode.appendChild(doc.createTextNode(text['label']))

        deletedNode = doc.createElement("deleted")
        deletedNode.appendChild(doc.createTextNode("0"))

        verifiedNode = doc.createElement("verified")
        verifiedNode.appendChild(doc.createTextNode("0"))

        occludedNone = doc.createElement("occluded")
        occludedNone.appendChild(doc.createTextNode("no"))

        dateNode = doc.createElement("date")

        idNode = doc.createElement("id")
        idNode.appendChild(doc.createTextNode(str(i)))

        partsNode = doc.createElement("parts")
        haspartsNode = doc.createElement("hasparts")
        ispartof = doc.createElement("ispartof")
        partsNode.appendChild(haspartsNode)
        partsNode.appendChild(ispartof)

        typeNode = doc.createElement("type")
        typeNode.appendChild(doc.createTextNode("bounding_box"))

        polygonNode = doc.createElement("polygon")
        for j in text["pos"]:
            ptNode = doc.createElement("pt")
            xNode = doc.createElement("x")
            xNode.appendChild(doc.createTextNode(str(j[0])))
            yNode = doc.createElement("y")
            yNode.appendChild(doc.createTextNode(str(j[1])))
            ptNode.appendChild(xNode)
            ptNode.appendChild(yNode)
            polygonNode.appendChild(ptNode)


        usernameNode = doc.createElement("username")
        usernameNode.appendChild(doc.createTextNode("cvat"))
        polygonNode.appendChild(usernameNode)

        attributesNode = doc.createElement("attributes")
        attributesNode.appendChild(doc.createTextNode("text="+text["content"]+"♥"+"entity="+text["entity"]))

        objectNode.appendChild(nameNode)
        objectNode.appendChild(deletedNode)
        objectNode.appendChild(verifiedNode)
        objectNode.appendChild(occludedNone)
        objectNode.appendChild(dateNode)
        objectNode.appendChild(idNode)
        objectNode.appendChild(partsNode)
        objectNode.appendChild(typeNode)
        objectNode.appendChild(polygonNode)
        objectNode.appendChild(attributesNode)

        root.appendChild(objectNode)

    fp = open(os.path.join(reserve_xmlpath,res_json["image_name"][:-4]+".xml"), 'w', encoding='utf-8')
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")



if __name__ == "__main__":
    pic_file = "../data/sroie_test_images"
    new_xml_path = "../data/sroie_test_xmls"
    pre_result = "../data/output_folder"
    origin_txt = "../data/sroie_test_box_and_transcripts"
    os.makedirs(new_xml_path, exist_ok=True)
    for file in os.listdir(pic_file):
        tem_info_dict = {}
        with open(os.path.join(origin_txt, file[:-4]+".txt")) as f1:
            data = f1.readlines()
            data = [item.strip() for item in data]
        with open(os.path.join(pre_result, file[:-4]+".txt")) as f2:
            pre_label_ = f2.readlines()
            pre_label_ = [item.strip().split('\t') for item in pre_label_]
        tem_info_dict['image_name'] = file
        img = cv2.imread(os.path.join(pic_file, file))
        # shape = img.shape
        tem_info_dict['imagesize'] = [str(img.shape[0]),str(img.shape[1])]
        tem_info_dict['text'] = []
        for index, line in enumerate(data):
            row = line.split(',', maxsplit=8)
            if len(row) != 9:
                print(data)
                print("$$$$", row, line, file)
            # xmin, ymin, xmax, ymax, Object, label
            tem_d = {}
            tem_d['pos'] = [(row[0],row[1]),(row[2],row[3]),
                            (row[4], row[5]),(row[6],row[7])]
            tem_d['content'] = row[-1]
            tem_d['entity'] = row[-1]
            tem_d['label'] = 'other'

            for attend_span in pre_label_:
                if (attend_span[1] == tem_d['content'] or attend_span[1] in tem_d['content']) \
                        and attend_span[0] != 'total':
                    tem_d['entity'] = attend_span[1]
                    tem_d['label'] = attend_span[0]

            tem_info_dict['text'].append(tem_d)
        # print(tem_info_dict)
        save_xml(tem_info_dict,new_xml_path)



