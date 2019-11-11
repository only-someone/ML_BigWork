#! -*- coding: utf-8 -*-


import os
from PIL import Image
import math
from utils.file_utils import create_if_not_exists, copy_file
from utils.xml_utils import create_xml_file
class PASCALVOC07(object):

    def __init__(self, trainval_anno, test_anno, val_ratio, out_dir, attrs):
        self._trainval_anno = trainval_anno
        self._test_anno = test_anno
        self._val_ratio = val_ratio
        self._out_dir = out_dir
        self._attrs = attrs

        self._jpegimages_dir = None
        self._imagesets_dir = None
        self._annotations_dir = None
        self._img_idx = 0

    def _build_voc_dir(self):
        self._out_dir = self._out_dir
        create_if_not_exists(os.path.join(self._out_dir, 'Annotations'))
        create_if_not_exists(os.path.join(self._out_dir, 'ImageSets'))
        create_if_not_exists(os.path.join(self._out_dir, 'ImageSets', 'Layout'))
        create_if_not_exists(os.path.join(self._out_dir, 'ImageSets', 'Main'))
        create_if_not_exists(os.path.join(self._out_dir, 'ImageSets', 'Segmentation'))
        create_if_not_exists(os.path.join(self._out_dir, 'JPEGImages'))
        create_if_not_exists(os.path.join(self._out_dir, 'SegmentationClass'))
        create_if_not_exists(os.path.join(self._out_dir, 'SegmentationObject'))
        self._annotations_dir = os.path.join(self._out_dir, 'Annotations')
        self._jpegimages_dir = os.path.join(self._out_dir, 'JPEGImages')
        self._imagesets_dir = os.path.join(self._out_dir, 'ImageSets', 'Main')

    def _create_annotation(self, image_idx, boxes, anno_file ,image_path,delimiter=' '):

        anno_file = os.path.join(self._annotations_dir, format(image_path[0:len(image_path)-4]+".xml"))

        attrs = dict()
        attrs['image_name'] = format(image_path)
        attrs['boxes'] = boxes
        img = Image.open(os.path.join(self._jpegimages_dir, format(image_path)))
        width, height = img.size
        attrs['width'] = str(width)
        attrs['height'] = str(height)
        for k, v in self._attrs.items():
            attrs[k] = v
        create_xml_file(anno_file, attrs)




    def _build_subset(self, start_idx, phase, anno_file, verbose=True, delimiter=' '):
        bili_map={}
        scale_map={}
        fout = open(os.path.join(self._imagesets_dir, '{}.txt'.format(phase)), 'w',encoding='utf8')
        n = 0
        with open(anno_file, 'r',encoding='utf8') as anno_f:
            for line in anno_f:
                line_split = line.strip().split(delimiter)

                # image saved path
                image_path = line_split[0]
                if verbose:
                    print("process img: {}".format(image_path))
                img = Image.open(os.path.join(self._jpegimages_dir, format(image_path)))
                width, height = img.size
                # a ground truth with bounding box
                boxes = []
                for i in range(int((len(line_split) - 1) / 5)):
                    category = line_split[1 + i * 5 + 0]
                    if category=="不带电芯充电宝":
                        category="coreless"
                    elif category=="带电芯充电宝":
                        category="core"
                    else:
                        continue
                    if int(line_split[1 + i * 5 + 1])<width:
	                    x1 = str(max(10,int(line_split[1 + i * 5 + 1])))
	                    y1 = str(max(10,int(line_split[1 + i * 5 + 2])))
	                    x2 = str(max(int(x1)+1,min(width-1,int(line_split[1 + i * 5 + 3]))))
	                    y2 = str(max(int(y1)+1,min(height-1,int(line_split[1 + i * 5 + 4]))))
	                    box_width=int(x2)-int(x1)
	                    box_height=int(y2)-int(y1)
	                    bili=round(max(box_height/box_width,box_width/box_height),2)
	                    scale=round(box_width/16,1)
	                    boxes.append((category, x1, y1, x2, y2))
	                    if not bili in bili_map:
	                        bili_map[bili]=1
	                    else:
	                        bili_map[bili]=1+bili_map[bili]
	                    if not scale in scale_map:
	                        scale_map[scale]=1
	                    else:
	                        scale_map[scale]=1+scale_map[scale]
	                  
                image_idx = start_idx + n
                n += 1
                # copy and rename image by index number
                #copy_file(image_path, self._jpegimages_dir, format(image_path))

                # write image idx to imagesets file
                fout.write(format(image_path[0:len(image_path)-4]) + '\n')

                # create annotation file
                self._create_annotation(image_idx, boxes,anno_file,image_path,delimiter=' ') 
        bili_map=sorted(bili_map.items(),key=lambda x:x[0])
        scale_map=sorted(scale_map.items(),key=lambda x:x[0])  
       	#print(bili_map)
        with open("bili.txt","w")as f:
            for i in bili_map:
                f.write(str(i[0])+",")
            f.write("\n") 
            for i in bili_map:
                f.write(str(i[1])+",") 
              
        with open("scale.txt","w")as f:
            for i in scale_map:
                f.write(str(i[0])+",")
            f.write("\n") 
            for i in scale_map:
                f.write(str(i[1])+",")
        return n

    def build(self, start_idx=1, verbose=True):
        self._build_voc_dir()
        
        n = self._build_subset(start_idx, "trainval", self._trainval_anno, verbose)
        self._build_subset(n + start_idx, "test", self._test_anno, verbose)
