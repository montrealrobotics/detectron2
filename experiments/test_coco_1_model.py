import random
import os
import glob 
import cv2
import numpy as np
import torch
import json
from detectron2.structures import BoxMode
import itertools
import sys
# import some common detectron2 utilities
import pdb
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from coco_custom_load import load_coco_json
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode

class_list = ['object']
json_file_train = "/network/tmp1/bhattdha/coco/annotations/instances_train2017.json"
image_root_train = "/network/tmp1/bhattdha/coco/train2017/"


DatasetCatalog.register("coco_1_class", lambda d='train': load_coco_json(json_file_train, image_root_train))
MetadataCatalog.get("coco_1_class").set(thing_classes=class_list)

coco_1_class_metadata = MetadataCatalog.get('coco_1_class')



cfg = get_cfg()
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_R_152_FPN_3x.yaml")


# loading config used during train time
# cfg_dict = torch.load('/network/tmp1/bhattdha/detectron2_kitti/resnet-50_FPN/resnet-50_FPN_cfg.final')
# cfg_dict = torch.load('/network/tmp1/bhattdha/detectron2_coco/coco_1_class_deterministic_maskrcnn/coco_1_class_deterministic_maskrcnn_cfg.final')
# cfg_dict = torch.load('/network/tmp1/bhattdha/detectron2_coco/coco_1_class_deterministic/coco_1_class_deterministic_cfg.final')
cfg_dict = torch.load('/network/tmp1/bhattdha/detectron2_coco/coco_1_class_more_layers_stage_2/coco_1_class_more_layers_stage_2_cfg.final')
cfg = cfg_dict['cfg']

# cfg.DATASETS.TRAIN = ("kitti/train",)
cfg.DATASETS.TEST = ('coco_2017_val',)   # no metrics implemented for this dataset


cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)  #  (kitti)

cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_coco/coco_1_class_more_layers_stage_2'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0099999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set the testing threshold for this model
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
cfg.MODEL.MASK_ON = False
predictor = DefaultPredictor(cfg)


all_image_paths = glob.glob('/network/home/bhattdha/tensorflow_datasets/downloads/extracted/ZIP.dhbw-stuttgar.de_sgehrig_lostAndF_leftImg8MH9mACAjq1l9MJljuUmQ9bmo5XNe5ynDKSZHpm6fKxg.zip/leftImg8bit/train/*/*.png')
for ind, image_path in enumerate(all_image_paths):
    print("On image: ", image_path)
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=coco_1_class_metadata, 
                   scale=1.0, 
                   instance_mode=ColorMode.IMAGE   
    )

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('/network/tmp1/bhattdha/detectron2_coco/coco_1_class_more_layers_stage_2/laf_out/' + str(ind).zfill(6) + '.png', v.get_image()[:, :, ::-1]) 
