"""
bbox计数器，用于统计各个类别的数量，适用于 Pascal VOC 数据集。
使用样例：
python vocbbox_counter.py -a ./Annotations \
    -i ./ImageSets/Main/test.txt \
    -p normal

normal模式下的输出，
    类名           : 该类bbox数量 

    sgi-nds 200    :   96
    dell-r710      :  100
    nf5270m3       :   66
    imp-p550       :   14
    dell-r720      :   34
    bbox class num :    5
    bbox total num :  310

detail模式下的输出，
    key:  类名
             序号,    文件名1,  包含该类bbox的个数
             序号,    文件名2,  包含该类bbox的个数
             ...
    类名 bbox sum:   该类bbox的总数
    
    key:  imp-p550
             1,    003171,  1
             2,    003191,  1
             3,    003281,  1
             4,    003761,  1
             5,    003771,  1
             6,    003781,  1
             7,    003791,  1
             8,    003801,  1
             9,    003811,  1
            10,    003821,  1
            11,    003831,  1
            12,    003841,  1
            13,    003851,  1
            14,    003861,  1
    imp-p550 bbox sum:   14
--------------------------
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import argparse
import collections

class VocBboxCounter():
    def __init__(self):
        self.annopath = ""
        self.txt_list = ""
        self.pattern = "normal"

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
    def bbox_counter(self):
        dic = {}  
        # 用于保存结果
        # 例如 { 'sgi-nds 200': 95, 
        #        'dell-r710': 101, 
        #        'nf5270m3': 66, 
        #        'imp-p550': 14, 
        #        'dell-r720': 34
        #      }
        detail_dic = {}  
        # 用于保存详细结果
        # { 类名1  :  [(文件名1, 该类bbox个数), (文件名2, 该类bbox个数)]
        #   类名2  :  [(文件名1, 该类bbox个数), (文件名4, 该类bbox个数)]
        # }   
        # 例如 {'sgi-nds 200': [('000000', 3), ('000001', 1)], 
        #       'dell-r710': [('000002', 2), ('000003', 2)] 
        #      }
        for boxes, clss, xml_url in self.bbox_gen():
            assert len(boxes) == len(clss)
            for b, c in zip(boxes, clss):
                if not c in dic:
                    dic[c] = 1
                else:
                    dic[c] += 1

            fname = os.path.splitext(os.path.split(xml_url)[1])[0]
            clss_d = list(set(clss))
            d = collections.Counter(clss) #例如 Counter({'eyes': 8, 'the': 5, 'look': 4})
            for c in clss_d:
                num = d[c]
                if not c in detail_dic:
                    detail_dic[c] = [(fname, num)]
                else:
                    detail_dic[c].append((fname, num))
        return dic, detail_dic

    def start(self):
        dic, detail_dic = self.bbox_counter()  
        if self.pattern == "normal":
            for key, v in dic.items():
                print("{:15}:{:>5}".format(key, v))
        elif self.pattern == "detail":
            for key, value in detail_dic.items():
                print("key: ", key)
                for i, (name, num) in enumerate(value):
                    print("{:>10},{:>10},{:>3}".format(i+1, name, num))
                print("{} bbox sum:{:>5}".format(key, dic[key]))
                print("--------------------------")
        else:
            raise Exception("not support pattern: ", self.pattern)

        print("{:15}:{:>5}".format("bbox class num", len(dic.keys())))
        print("{:15}:{:>5}".format("bbox total num", sum(dic.values())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bbox counter for Pascal VOC dataset")
    parser.add_argument('-a','--anno', type=str, required=True, help="annotation path")
    parser.add_argument('-i','--input', type=str, required=True, help="path of imageset txt")
    parser.add_argument('-p','--pattern', type=str, default="normal", help="several pattern support: 'normal' for brevity, 'detail' for comprehensive results")
    args = parser.parse_args()
    print(args)

    bc = VocBboxCounter()
    bc.annopath = args.anno
    bc.txt_list = args.input
    bc.pattern = args.pattern
    bc.start()


