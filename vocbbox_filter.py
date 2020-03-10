class bbox_filter(object):
    """
    Load bounding boxes info from XML file in the PASCAL VOC
    format, and filtering
    """
    self.op_func = None

    # lt：less than 小于
    # le：less than or equal to 小于等于
    # eq：equal to 等于
    # ne：not equal to 不等于
    # ge：greater than or equal to 大于等于
    # gt：greater than 大于
    def op_func_le(self, v1, v2):
        return v1 <= v2

    def filt(self, boxes, key, op_func, v):
        dic = {}
        ret = False
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
            ret = op_func(dic[key], v)
            if ret:
                break
        return ret

    def bbox_reader(self, filename):
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        return boxes, gt_classes

    def start(self, txt_list, xmlpath, out_txt):
        with open(out_txt, "w") as f_out:
            with open(txt_list, "r") as f:
                for line in f:
                    name = line.strip()
                    file_url= os.path.join(xmlpath, name+".xml")
                    boxes, gt_classes = self.bbox_reader(file_url)
                    ret = filt(boxes, key, self.op_func, v)
                    if ret:
                        f_out.write(name)

def mkr(path):
    if os.path.exists(path):
        print("path already exists: {}, remove it".format(path))
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        print("path not exists: {}, make it".format(path))
        os.makedirs(path)
            
if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="bbox filter")
#    parser.add_argument('-i','--imgs', type=str, help="input path of image path")
#    parser.add_argument('-a','--annos', type=str, help="input path of annotation path")
#    parser.add_argument('-o','--outtxt', type=str, help="output txt")
#    args = parser.parse_args()
#    print(args)
#    filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
#    out_img_path = os.path.join(args.outpath, os.path.split(args.imgs)[1])
#    out_anno_path = os.path.join(args.outpath, os.path.split(args.annos)[1])
    txt_list = "/share/faster_rcnn-bk/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt"
    xmlpath = "/share/faster_rcnn-bk/data/VOCdevkit2007/VOC2007/Annotations"
    out_txt = os.path.join(os.getcwd(), "out.txt")
    mkr(out_txt)
    bf = bbox_filter()
    bf.op_func = bf.op_func_le
    bf.start(txt_list, xmlpath, out_txt)


