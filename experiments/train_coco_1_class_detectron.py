
"""
In this code, we are training a model with 1 class for all coco 
classes.

All 80 categories of coco dataset are mapped as "object"
category.

Purpose:

To see if such variations enable the object detector to 
learn generic concept of object or not! After training, 
this should be tested for dataset like Kitti/Cityscapes.

"""


import random
import os
import glob 
import cv2
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import sys
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from contextlib import redirect_stdout
from coco_custom_load import load_coco_json
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-name", "--experiment_comment", required = True, help="Comments for the experiment")

args = vars(ap.parse_args())

dir_name = args['experiment_comment']

## only 1 class! 
class_list = ['object']

from detectron2.data import DatasetCatalog, MetadataCatalog


json_file_train = "/network/tmp1/bhattdha/coco/annotations/instances_train2017.json"
image_root_train = "/network/tmp1/bhattdha/coco/train2017/"


DatasetCatalog.register("coco_1_class", lambda d='train': load_coco_json(json_file_train, image_root_train))
MetadataCatalog.get("coco_1_class").set(thing_classes=class_list)

coco_1_class_metadata = MetadataCatalog.get('coco_1_class')


print("data loading")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("/home/mila/b/bhattdha/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_1_class.yaml")
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_deform_conv_3x.yaml")
cfg.DATASETS.TRAIN = ("coco_1_class",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # initialize from model zoo
cfg.MODEL.WEIGHTS = "/home/mila/b/bhattdha/model_final_a3ec72.pkl"
# cfg.MODEL.WEIGHTS = "/network/tmp1/bhattdha/detectron2_kitti/model_0014999.pth"  # initialize fron deterministic model
cfg.SOLVER.IMS_PER_BATCH = 5
# cfg.SOLVER.BASE_LR = 0.015
cfg.SOLVER.BASE_LR = 1e-3
cfg.SOLVER.MAX_ITER =  250000  
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)  #  (kitti)
cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_coco/' + dir_name
# cfg.CUSTOM_OPTIONS.DETECTOR_TYPE = 'deterministic'
# cfg.CUSTOM_OPTIONS.STRUCTURED_EDGE_RESPONSE = True

cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.8

if cfg.STRUCTURED_EDGE_RESPONSE.ENABLE:
    cfg.MODEL.PIXEL_MEAN = cfg.STRUCTURED_EDGE_RESPONSE.PIXEL_MEAN[cfg.STRUCTURED_EDGE_RESPONSE.INPUT_TYPE]
    cfg.MODEL.PIXEL_STD = cfg.STRUCTURED_EDGE_RESPONSE.PIXEL_STD[cfg.STRUCTURED_EDGE_RESPONSE.INPUT_TYPE]

if cfg.CUSTOM_OPTIONS.DETECTOR_TYPE is 'deterministic':
    ## has to be smooth l1 loss if detector is deterministc
    cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'smooth_l1'


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
### At this point, we will save the config as it becomes vital for testing in future
torch.save({'cfg': cfg}, cfg.OUTPUT_DIR + '/' + dir_name + '_cfg.final')

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
print("start training")
print("The checkpoint iteration value is: ", cfg.SOLVER.CHECKPOINT_PERIOD)
trainer.train()


