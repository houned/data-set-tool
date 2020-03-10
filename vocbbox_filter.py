"""
bbox滤波器，用于过滤 Pascal VOC 数据集。
使用样例：
python vocbbox_filter.py -a ./Annotations \
    -i ./ImageSets/Main/trainval.txt \
    -o ./out.txt \
    -t 3 \
    -k width \
    -f le
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import argparse

class bbox_filter():
    def __init__(self):
        self.annopath = ""
        self.txt_list = ""
        self.out_txt = ""
        self.op_func = None
        self.key = ""
        self.thred = 19

    # lt：less than 小于
    # le：less than or equal to 小于等于
    # eq：equal to 等于
    # ne：not equal to 不等于
    # ge：greater than or equal to 大于等于
    # gt：greater than 大于
    def op_func_le(self, v1, v2):
        return v1 <= v2

    #测试集xml路径生成器
    def dataset_gen(self):
        with open(self.txt_list) as f:
            lines = f.readlines()
        for name in lines:
            yield os.path.join(self.annopath, name.strip()+".xml")

    #bbox生成器
    def bbox_gen(self):
        for xml_url in self.dataset_gen():
            tree = ET.parse(xml_url)
            objs = tree.findall('object')
            num_objs = len(objs)

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)

            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)

                cls = obj.find('name').text.lower().strip()  #类别名
                boxes[ix, :] = [x1, y1, x2, y2]
            yield boxes, xml_url
        
    #条件滤波器
    def condition_filter(self):
        dic = {}
        for boxes, xml_url in self.bbox_gen():
            isChosen = False
            for b in boxes:
                xmin = b[0]
                ymin = b[1]
                xmax = b[2]
                ymax = b[3]
                dic["xmin"] = xmin
                dic["ymin"] = ymin
                dic["xmax"] = xmax
                dic["ymax"] = ymax
                dic["width"] = xmax - xmin +1
                dic["height"] = ymax - ymin +1
                isChosen = self.op_func(dic[self.key], self.thred)
                if isChosen:
                    print("{:4} is less or equal than thred".format(dic[self.key]))
                    break
            if isChosen:
                yield xml_url

    def start(self):
        with open(self.out_txt, "w") as f:
            for path in self.condition_filter():
                basename = os.path.split(path)[1]
                f.write(os.path.splitext(basename)[0] + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bbox filter for filtering Pascal VOC dataset")
    parser.add_argument('-a','--anno', type=str, help="annotation path")
    parser.add_argument('-i','--input', type=str, help="path of imageset txt")
    parser.add_argument('-o','--output', default="./out.txt", type=str, help="path of output txt")
    parser.add_argument('-t','--thred', type=int, help="filter threshold")
    parser.add_argument('-k','--key', type=str, help="keyword support: 'xmin','ymin','xmax','ymax','width', 'height'")
    parser.add_argument('-f','--func', type=str, help="filter func support: 'lt','le','eq','ne','ge', 'gt'")
    args = parser.parse_args()
    print(args)

    
    bf = bbox_filter()
    bf.annopath = args.anno
    bf.txt_list = args.input
    bf.out_txt = args.output
    bf.key = args.key
    bf.thred = args.thred
    if args.func == "le":
        bf.op_func = bf.op_func_le
    else:
        print("not support filter func '{}'".format(args.func))
    # bf.annopath = "/share/faster_rcnn-bk/data/VOCdevkit2007/VOC2007/Annotations"
    # bf.txt_list = "/share/faster_rcnn-bk/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt"
    # bf.out_txt = os.path.join(os.getcwd(), "out.txt")
    # bf.op_func = bf.op_func_le
    # bf.key = "width"
    # bf.thred = 3
    bf.start()


