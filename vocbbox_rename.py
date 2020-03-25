"""
bbox更名器，用于更改bbox的name属性，适用于 Pascal VOC 数据集。
使用样例：
python vocbbox_rename.py -a ./Annotations \
    -i ./ImageSets/Main/trainval.txt \
    -o ./out \
    -t imp-p550 \
    -c ibm-p550
"""
import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import argparse
import collections

class VocBboxCounter():
    def __init__(self):
        self.annopath = ""
        self.txt_list = ""
        self.outpath = ""
        self.target = ""
        self.change = ""

    #测试集xml路径生成器
    def dataset_gen(self):
        with open(self.txt_list) as f:
            lines = f.readlines()
        for name in lines:
            yield os.path.join(self.annopath, name.strip()+".xml")

    #xml内容生成器
    def tree_gen(self):
        for xml_url in self.dataset_gen():
            tree = ET.parse(xml_url)
            objs = tree.findall('object')
            # num_objs = len(objs)

            # 遍历'object'节点
            is_choosed = False
            for ix, obj in enumerate(objs):
                n_node = obj.find('name')
                name = n_node.text.lower().strip()  #类别名
                if name == self.target:
                    n_node.text = self.change
                    is_choosed = True    
            if is_choosed:
                yield tree, os.path.split(xml_url)[1]

    def start(self):
        count = 0
        for tree, fname in self.tree_gen():
            outxml = os.path.join(self.outpath, fname)
            tree.write(outxml)
            print("{} changed: {} ---> {}".format(fname, self.target, self.change))
            count += 1
        print("{} xml files changed, save to {}".format(count, self.outpath))

def mkr(path):
    if os.path.exists(path):
        print("path already exists: {}, remove it".format(path))
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        print("path not exists: {}, make it".format(path))
        os.makedirs(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bbox counter for Pascal VOC dataset")
    parser.add_argument('-a','--anno', type=str, required=True, help="annotation path")
    parser.add_argument('-i','--input', type=str, required=True, help="path of imageset txt")
    parser.add_argument('-o','--outpath', type=str, required=True, help="changed annotation save in this path")
    parser.add_argument('-t','--target', type=str, required=True, help="target bbox name")
    parser.add_argument('-c','--change', type=str, required=True, help="new name to change to")
    args = parser.parse_args()
    print(args)

    mkr(args.outpath)

    bc = VocBboxCounter()
    bc.annopath = args.anno
    bc.txt_list = args.input
    bc.outpath = args.outpath
    bc.target = args.target
    bc.change = args.change
    bc.start()


