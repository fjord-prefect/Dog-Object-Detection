#!/usr/bin/env python

from pycocotools.coco import COCO
from tqdm import tqdm
import os
import pylab
import numpy as np
import skimage.io as io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="data type from annotation folder train2017 or val2017",
                    type=str)
args = parser.parse_args()
DATATYPE = args.data

def dataset(dataType):
    dir_=dataType[:-4]
    annFile='./annotations/instances_{}.json'.format(dataType)
    coco=COCO(annFile)

    catIds = coco.getCatIds(catNms=['dog']);

    imgIds = coco.getImgIds(catIds=catIds );

    for id_ in tqdm(imgIds):
        try:
            coco_data = coco.loadImgs(id_)[0]
            coco_bbox = coco.loadAnns(coco.getAnnIds(id_, catIds=catIds, iscrowd=None))[0]['bbox']
            yolo_bbox = [coco_bbox[0], coco_bbox[1], coco_bbox[2] + coco_bbox[0], coco_bbox[3] + coco_bbox[1]]
            str_bbox = ','.join([str(round(i)) for i in yolo_bbox])
            pic = io.imread(coco_data['coco_url'])
            path = os.path.join('data', dir_, '{}.jpg'.format(str(id_)))
            anno = path + ' ' + str_bbox + ',0'
            io.imsave(path, pic)
    
            with open('data/{}.txt'.format(dir_), 'a') as writer:
                writer.write('%s\n' % anno)
        except:
            print('missed one')

    return None

dataset(DATATYPE)

