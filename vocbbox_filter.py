"""
bbox滤波器，用于过滤 Pascal VOC 数据集。
使用样例1：
以bbox宽度为滤波条件
python vocbbox_filter.py -a ./Annotations \
    -i ./ImageSets/Main/trainval.txt \
    -o ./out.txt \
    -t 3 \
    -k width \
    -f le \
    -p any
使用样例2：
以bbox name为滤波条件
python vocbbox_filter.py -a ./Annotations \
    -i ./ImageSets/Main/trainval.txt \
    -o ./out.txt \
    -t ibm-p550 \
    -k name \
    -f eq \
    -p any
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
        self.pattern = "any"

    # lt：less than 小于
    # le：less than or equal to 小于等于
    # eq：equal to 等于
    # ne：not equal to 不等于
    # ge：greater than or equal to 大于等于
    # gt：greater than 大于
    def op_func_lt(self, v1, v2):
        return v1 < v2
    
    def op_func_le(self, v1, v2):
        return v1 <= v2

    def op_func_eq(self, v1, v2):
        return v1 == v2

    def op_func_ne(self, v1, v2):
        return v1 != v2

    def op_func_ge(self, v1, v2):
        return v1 >= v2

    def op_func_gt(self, v1, v2):
        return v1 > v2

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
            clss = [None]*num_objs

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
                clss[ix] = cls
            yield boxes, clss, xml_url
        
    #条件滤波器
    def condition_filter(self):
        dic = {}
        for boxes, clss, xml_url in self.bbox_gen():
            assert len(boxes) == len(clss)
            isChosen = []
            chosen_val_lst = []
            for b, n in zip(boxes, clss):
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
                dic["name"] = n
                ret = self.op_func(dic[self.key], self.thred)
                isChosen.append(ret)
                if ret:
                    chosen_val_lst.append(dic[self.key])
            if self.pattern == "any":
                filt_res = True in isChosen
            elif self.pattern == "all":
                filt_res = not( False in isChosen )
            else:
                print("not support pattern:", self.pattern)
                assert self.pattern == "any" or self.pattern == "all"

            if filt_res:
                print("{:20} is chosen, which {} is {}".format(
                    os.path.splitext(os.path.split(xml_url)[1])[0],
                    self.key,
                    chosen_val_lst)
                    )
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
    parser.add_argument('-o','--output', type=str, default="./out.txt", help="path of output txt")
    parser.add_argument('-t','--thred', type=str, help="filter threshold")
    parser.add_argument('-k','--key', type=str, help="keyword support: 'xmin','ymin','xmax','ymax','width','height','name'")
    parser.add_argument('-f','--func', type=str, help="filter func support: 'lt','le','eq','ne','ge', 'gt'")
    parser.add_argument('-p','--pattern', type=str, default="any", help="filter pattern support: 'all' for all match,'any' for any match")
    args = parser.parse_args()
    print(args)

    
    bf = bbox_filter()
    bf.annopath = args.anno
    bf.txt_list = args.input
    bf.out_txt = args.output
    bf.key = args.key
    bf.thred = args.thred if args.key == "name" else float(args.thred)
    bf.pattern = args.pattern
    if args.func == "lt":
        bf.op_func = bf.op_func_lt
    elif args.func == "le":
        bf.op_func = bf.op_func_le
    elif args.func == "eq":
        bf.op_func = bf.op_func_eq
    elif args.func == "ne":
        bf.op_func = bf.op_func_ne
    elif args.func == "ge":
        bf.op_func = bf.op_func_ge
    elif args.func == "gt":
        bf.op_func = bf.op_func_gt
    else:
        raise Exception("not support filter func '{}'".format(args.func))
    if args.key == "name" and args.func != "eq":
        raise Exception("key 'name' only support filter func '{}'".format(args.func))
    # bf.annopath = "/share/faster_rcnn-bk/data/VOCdevkit2007/VOC2007/Annotations"
    # bf.txt_list = "/share/faster_rcnn-bk/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt"
    # bf.out_txt = os.path.join(os.getcwd(), "out.txt")
    # bf.op_func = bf.op_func_le
    # bf.key = "width"
    # bf.thred = 3
    bf.start()


