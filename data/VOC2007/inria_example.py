#! -*- coding: utf-8 -*-

import os
import sys
from easydict import EasyDict as edict

from pascal_voc.pascal_voc import PASCALVOC07

config = edict()

config.author = "anonymous"
config.root = "annotation"
config.folder = "VOC2007"
config.annotation = "PASCAL VOC2007"
config.segmented = "0"
config.difficult = "0"
config.truncated = "0"
config.pose = "Unspecified"
config.database = "The VOC2007 Database"
config.depth = "3"


if __name__ == "__main__":

    inria_dir = "./"
    out_dir = "./"

    trainval_anno = os.path.join(inria_dir, '', '测试集和训练集/Train_annotation.txt')
    test_anno = os.path.join(inria_dir, '', '测试集和训练集/Test_annotation.txt')

    val_ratio = 0

    p = PASCALVOC07(trainval_anno, test_anno, val_ratio, out_dir, config)
    p.build(True)
