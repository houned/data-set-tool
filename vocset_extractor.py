"""
数据集抽取器
从数据集中，均匀抽取数据，组成测试集，文件名输出至文件。
特别适用于从视频中抽取的数据集。
使用样例：
1.均匀抽取器
python vocset_extractor.py -i ls.txt \
      -oe ext.txt -or res.txt -p uniform -v 10
"""
import os
import numpy as np
import argparse

class VOCSetExtractor():
    def __init__(self):
        self.txt_list = ""
        self.outtxt_ext = "ext.txt"
        self.outtxt_res = "res.txt"
        self.pattern = ""
        self.interval = 1

    # jpg文件名生成器
    # def filename_gen(self):
    #     for name in os.listdir(self.jpg_path):
    #         if len(name) > 4:
    #             if name[-4:] == ".jpg":
    #                 yield name[:-4]

    # jpg文件名生成器
    def filename_gen(self):
        with open(self.txt_list) as f:
            lines = f.readlines()
        for name in lines:
            yield name.strip()

    # 分路器-均匀分路器
    def uniform_extractor(self):
        intv = 0
        for name in self.filename_gen():
            if intv == 0:
                intv = self.interval 
                yield name, "ext"
            else:
                yield name, "res"
            intv -= 1

    def start(self):
        with open(self.outtxt_ext, "w") as fe:
            with open(self.outtxt_res, "w") as fr:
                if self.pattern == "uniform":
                    for name, setn in self.uniform_extractor():
                        if setn == "ext":
                            fe.write(name + "\n")
                        elif setn == "res":
                            fr.write(name + "\n")
                elif self.pattern == "rand":
                    print("code empty")
                    pass
                else:
                    raise Exception("not support pattern: {}".format(self.pattern))
        print("work done")

# 删除存在的文件
def delfile(filepath):
    if os.path.exists(filepath):
        print("file exist, remove it ", filepath)
        os.remove(filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="set extractor for Pascal VOC dataset")
    parser.add_argument('-i','--input', type=str, help="path of imageset txt")
    parser.add_argument('-oe','--outtxt_ext', type=str, default="./ext.txt", help="path of extractor output txt")
    parser.add_argument('-or','--outtxt_res', type=str, default="./res.txt", help="path of residuals output txt")
    parser.add_argument('-p','--pattern', type=str, default="uniform", help="pattern of extractor, support: 'uniform' for uniform extractor, 'rand' for random extractor")
    parser.add_argument('-v','--interval', type=int, default=10, help="interval of extractor")
    args = parser.parse_args()
    print(args)

    delfile(args.outtxt_ext)
    delfile(args.outtxt_res)
    
    se = VOCSetExtractor()
    se.txt_list = args.input
    se.outtxt_ext = args.outtxt_ext
    se.outtxt_res = args.outtxt_res
    se.pattern = args.pattern
    se.interval = args.interval
    
    # bf.annopath = "/share/faster_rcnn-bk/data/VOCdevkit2007/VOC2007/Annotations"
    # bf.txt_list = "/share/faster_rcnn-bk/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt"
    # bf.out_txt = os.path.join(os.getcwd(), "out.txt")
    # bf.op_func = bf.op_func_le
    # bf.key = "width"
    # bf.thred = 3
    se.start()