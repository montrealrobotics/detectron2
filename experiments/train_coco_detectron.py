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
from fvcore.common.file_io import PathManager

from detectron2.config import get_cfg
import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from contextlib import redirect_stdout
from coco_custom_load import load_coco_json
import argparse

# os.environ["DETECTRON2_DAT"] = "my_export"

ap = argparse.ArgumentParser()
ap.add_argument("-name", "--experiment_comment", required = True, help="Comments for the experiment")

args = vars(ap.parse_args())

dir_name = args['experiment_comment']

## only 1 class! 
# class_list = ['object']

from detectron2.data import DatasetCatalog, MetadataCatalog


# json_file_train = "/network/tmp1/bhattdha/coco/annotations/instances_train2017.json"
# image_root_train = "/network/tmp1/bhattdha/coco/train2017/"


# DatasetCatalog.register("coco_1_class", lambda d='train': load_coco_json(json_file_train, image_root_train))
# MetadataCatalog.get("coco_1_class").set(thing_classes=class_list)

# coco_1_class_metadata = MetadataCatalog.get('coco_1_class')


print("data loading")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
# cfg.merge_from_file("/home/mila/b/bhattdha/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.merge_from_file("/home/mila/b/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_deform_conv_3x.yaml")


cfg.SOLVER.IMS_PER_BATCH = 24
cfg.SOLVER.BASE_LR = 1e-3
# cfg.SOLVER.STEPS: (210000, 250000)
cfg.SOLVER.MAX_ITER = 40000

cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.CUSTOM_OPTIONS.LOSS_WEIGHTS = [0.8, 0.2] 
# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # initialize from model zoo
# cfg.MODEL.WEIGHTS = "/home/mila/b/bhattdha/model_final_f6e8b1.pkl"
cfg.MODEL.WEIGHTS = "/network/tmp1/bhattdha/detectron2_coco/loss_att_unfrozen_uncert_head/model_0009999.pth"
cfg.SOLVER.CHECKPOINT_PERIOD = 2500  
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.CUSTOM_OPTIONS.RPN_FORGROUND_LOSS_ONLY = False
cfg.CUSTOM_OPTIONS.CORRUPT_BG = False
cfg.STRUCTURED_EDGE_RESPONSE.ENABLE = False
# cfg.CUSTOM_OPTIONS.ANNEALING_ITERATIONS = 5000
# cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'mahalanobis_attenuation'
# cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'smoothl1'
cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'kl_div_chi_sq_closed_form_plus_smoothl1'
cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_coco/' + dir_name
cfg.CUSTOM_OPTIONS.DETECTOR_TYPE = 'probabilistic'
# cfg.CUSTOM_OPTIONS.STRUCTURED_EDGE_RESPONSE = True

# cfg.CUSTOM_OPTIONS.DETECTOR_TYPE = 'deterministic'
cfg.CUSTOM_OPTIONS.ENCODING_TYPE = 'xyxy'



if cfg.STRUCTURED_EDGE_RESPONSE.ENABLE:
    cfg.MODEL.PIXEL_MEAN = cfg.STRUCTURED_EDGE_RESPONSE.PIXEL_MEAN[cfg.STRUCTURED_EDGE_RESPONSE.INPUT_TYPE]
    cfg.MODEL.PIXEL_STD = cfg.STRUCTURED_EDGE_RESPONSE.PIXEL_STD[cfg.STRUCTURED_EDGE_RESPONSE.INPUT_TYPE]

if cfg.CUSTOM_OPTIONS.DETECTOR_TYPE is 'deterministic':
    ## has to be smooth l1 loss if detector is deterministc
    cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'smooth_l1'

# cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
# cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.00005, 0.95]
# cfg.MODEL.ROI_HEADS.IOU_LABELS = [0, -1, 1]
# cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.99
# cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
### At this point, we will save the config as it becomes vital for testing in future
torch.save({'cfg': cfg}, cfg.OUTPUT_DIR + '/' + dir_name + '_cfg.final')
with PathManager.open(cfg.OUTPUT_DIR + '/' + dir_name + '_cfg.yaml', "w") as f:
	f.write(cfg.dump())

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
print("start training")
print("The checkpoint iteration value is: ", cfg.SOLVER.CHECKPOINT_PERIOD)

uncertainty_heads_name = ['roi_heads.box_predictor.bbox_uncertainty_pred.weight',
'roi_heads.box_predictor.bbox_uncertainty_pred.bias']

for name, p in trainer.model.named_parameters():
	if 'roi_heads' not in name:
		print(name)
		p.requires_grad = False
	# if name not in uncertainty_heads_name:
	# 	print(name)
	# 	p.requires_grad = False

trainer.train()


