"""
bbox检验器，用于检验 Pascal VOC 数据集是否合规。
当发现不合规，图片名输出至文件。
使用样例：
python vocbbox_checker.py -a ./Annotations \
    -i ./ImageSets/Main/trainval.txt \
    -o ./out.txt \
    -w 2 \
    -hm 2 \
    -e 1
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import argparse

class bbox_checker():
    def __init__(self):
        self.annopath = ""
        self.txt_list = ""
        self.out_txt = ""
        self.wmin = 0
        self.hmin = 0
        self.edge_dist = 0

    def width_checker(self, xmin, xmax):
        xmax = int(xmax)
        xmin = int(xmin)
        return (xmax - xmin) >= self.wmin
    
    def height_checker(self, ymin, ymax):
        ymax = int(ymax)
        ymin = int(ymin)
        return (ymax - ymin) >= self.hmin

    def point_checker(self, x, y, width, height):
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        ret = True
        ret = ret and (x >= self.edge_dist)
        ret = ret and (width - x - 1 >= self.edge_dist)
        ret = ret and (y >= self.edge_dist)
        ret = ret and (height - y -1 >= self.edge_dist)
        return ret

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

    #图像信息生成器
    def picinfo_gen(self):
        for xml_url in self.dataset_gen():
            tree = ET.parse(xml_url)
            sz = tree.find('size')
            width = int(sz.find('width').text)
            height = int(sz.find('height').text)
            depth = int(sz.find('depth').text)
            yield width, height, depth, xml_url

    #bbox检验器
    def bbox_checker(self):
        for bbox, picinfo in zip(self.bbox_gen(), self.picinfo_gen()):
            boxes, xml_url = bbox
            width, height, depth, xml_url_2 = picinfo
            assert xml_url == xml_url_2
            isChosen = False
            #chosen_val_lst = []
            for b in boxes:
                xmin = b[0]
                ymin = b[1]
                xmax = b[2]
                ymax = b[3]
                isNorm = True
                if not self.point_checker(xmin, ymin, width, height):
                    isNorm = False
                    err = "point checher abn (min point)"
                elif not self.point_checker(xmax, ymax, width, height):
                    isNorm = False
                    err = "point checher abn (max point)"
                elif not self.width_checker(xmin, xmax):
                    isNorm = False
                    err = "width checher abn"
                elif not self.height_checker(ymin, ymax):
                    isNorm = False
                    err = "height checher abn"
                isChosen = isChosen or (not isNorm)
                if not isNorm:
                    print("{:20} is abnormal, pic size({:4},{:4}); with bbox {}, bbox size:({},{}), {}".format(
                        os.path.splitext(os.path.split(xml_url)[1])[0],
                        width,
                        height,
                        b,
                        xmax - xmin,
                        ymax - ymin,
                        err)
                        )
            if isChosen:
                yield xml_url

    def start(self):
        with open(self.out_txt, "w") as f:
            for path in self.bbox_checker():
                basename = os.path.split(path)[1]
                f.write(os.path.splitext(basename)[0] + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bbox checker for Pascal VOC dataset")
    parser.add_argument('-a','--anno', type=str, help="annotation path")
    parser.add_argument('-i','--input', type=str, help="path of imageset txt")
    parser.add_argument('-o','--output', type=str, default="./out.txt", help="path of output txt")
    parser.add_argument('-w','--wmin', type=int, default=0, help="min of bbox width")
    parser.add_argument('-hm','--hmin', type=int, default=0, help="min of bbox height")
    parser.add_argument('-e','--edgedst', type=int, default=0, help="min distance between bbox and border")
    args = parser.parse_args()
    print(args)

    bc = bbox_checker()
    bc.annopath = args.anno
    bc.txt_list = args.input
    bc.out_txt = args.output
    bc.wmin = args.wmin
    bc.hmin = args.hmin
    bc.edge_dist = args.edgedst
    
    # bf.annopath = "/share/faster_rcnn-bk/data/VOCdevkit2007/VOC2007/Annotations"
    # bf.txt_list = "/share/faster_rcnn-bk/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt"
    # bf.out_txt = os.path.join(os.getcwd(), "out.txt")
    # bf.op_func = bf.op_func_le
    # bf.key = "width"
    # bf.thred = 3
    bc.start()


