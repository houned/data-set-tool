# -*- coding: utf-8 -*-
# 为数据集的图片名和标注名添加前缀
# 用法示例
# python prefix_dataset.py -p prefix -i input_img_path -a input_annotation_path -o output_path
import os
import argparse
#if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        print("path already exists: {}, remove it".format(path))
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        print("path not exists: {}, make it".format(path))
        os.makedirs(path)
def prefixing(prefix, img_path, anno_path, out_img_path, out_anno_path):
    #jpeg_file_path = 'JPEGImages'
    #annotation_path = 'Annotations'
    total_jpeg = os.listdir(img_path)
    total_xml = os.listdir(anno_path)
    for i, srcname in enumerate(total_jpeg):
        name=os.path.splitext(srcname)[0]
        obj_xml = name+".xml"
        if not obj_xml in total_xml:
            print("error!", obj_xml, "miss")
        src_jpg = os.path.join(img_path, srcname)
        dst_jpg = os.path.join(out_img_path, prefix +"_"+ srcname)
        src_xml = os.path.join(anno_path, obj_xml)
        dst_xml = os.path.join(out_anno_path, prefix +"_"+ obj_xml)
        os.rename(src_jpg, dst_jpg)
        os.rename(src_xml, dst_xml) 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="add prefix on dataset pics")
    parser.add_argument('-p','--prefix', type=str, help="prefix before image-names")
    parser.add_argument('-i','--imgs', type=str, help="input path of image path")
    parser.add_argument('-a','--annos', type=str, help="input path of annotation path")
    parser.add_argument('-o','--outpath', type=str, help="output path")
    args = parser.parse_args()
    print(args)
    out_img_path = os.path.join(args.outpath, os.path.split(args.imgs)[1])
    out_anno_path = os.path.join(args.outpath, os.path.split(args.annos)[1])
    mkr(out_img_path)
    mkr(out_anno_path)
    prefixing(args.prefix, args.imgs, args.annos, out_img_path, out_anno_path)
