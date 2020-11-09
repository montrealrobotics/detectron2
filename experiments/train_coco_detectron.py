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
cfg.SOLVER.BASE_LR = 1e-4
# cfg.SOLVER.STEPS: (210000, 250000)
cfg.SOLVER.MAX_ITER = 30000

cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.CUSTOM_OPTIONS.LOSS_WEIGHTS = [1.0, 2.0] 	
# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # initialize from model zoo
# cfg.MODEL.WEIGHTS = "/home/mila/b/bhattdha/model_final_f6e8b1.pkl"
cfg.MODEL.WEIGHTS = "/network/tmp1/bhattdha/detectron2_coco/coco_loss_attenuation/model_0019999.pth"
cfg.SOLVER.CHECKPOINT_PERIOD = 2500
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.CUSTOM_OPTIONS.RPN_FORGROUND_LOSS_ONLY = False
cfg.CUSTOM_OPTIONS.CORRUPT_BG = False
cfg.STRUCTURED_EDGE_RESPONSE.ENABLE = False
cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_coco/' + dir_name
cfg.CUSTOM_OPTIONS.DETECTOR_TYPE = 'probabilistic'
cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'mahalanobis_attenuation'

# cfg.CUSTOM_OPTIONS.LEARN_RC_SHARPNESS = True
# cfg.CUSTOM_OPTIONS.NEW_UNCERTAINTY_HEAD = True

cfg.CUSTOM_OPTIONS.ENCODING_TYPE = 'xyxy'

if cfg.STRUCTURED_EDGE_RESPONSE.ENABLE:
    cfg.MODEL.PIXEL_MEAN = cfg.STRUCTURED_EDGE_RESPONSE.PIXEL_MEAN[cfg.STRUCTURED_EDGE_RESPONSE.INPUT_TYPE]
    cfg.MODEL.PIXEL_STD = cfg.STRUCTURED_EDGE_RESPONSE.PIXEL_STD[cfg.STRUCTURED_EDGE_RESPONSE.INPUT_TYPE]

if cfg.CUSTOM_OPTIONS.DETECTOR_TYPE is 'deterministic':
    ## has to be smooth l1 loss if detector is deterministc
    cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'smooth_l1'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
### At this point, we will save the config as it becomes vital for testing in future
torch.save({'cfg': cfg}, cfg.OUTPUT_DIR + '/' + dir_name + '_cfg.final')
with PathManager.open(cfg.OUTPUT_DIR + '/' + dir_name + '_cfg.yaml', "w") as f:
	f.write(cfg.dump())

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
print("start training")
print("The checkpoint iteration value is: ", cfg.SOLVER.CHECKPOINT_PERIOD)


uncertainty_heads_name = ['roi_heads.box_predictor.bbox_uncertainty_pred',
'roi_heads.box_predictor.bbox_uncertainty_pred']
regression_heads_name = ['roi_heads.box_predictor.bbox_pred.weight',
'roi_heads.box_predictor.bbox_pred.bias']

stage_1_info = 'unfrozen'
uncertainty_head = None

for name, p in trainer.model.named_parameters():
	if 'roi_heads' not in name:
		if stage_1_info is 'unfrozen':
			stage_1_info = 'frozen'
		print(name)
		p.requires_grad = False
	# if 'roi_heads.box_predictor.bbox_uncertainty_pred' not in name: # and name not in regression_heads_name:
	# 	if uncertainty_head is None:
	# 		uncertainty_head = 'unfrozen'
	# 	print(name)	
	# 	p.requires_grad = False
			
if cfg.CUSTOM_OPTIONS.LEARN_RC_SHARPNESS:
	trainer.model.roi_heads.box_predictor.sharpness.requires_grad = True

if uncertainty_head == 'unfrozen':
	model_info = 'This is a {} model being trained with {} loss. We are training it on {} dataset with batchsize {} and intial learning rate of {}. This model intialzes weights from {} file. We are training this model for {} iterations, which roughly translates into {} epochs. We have uncertainty_head unfrozen. Everything else is frozen. We store checkpoints at every {} iterations. In this model, existence of complex uncertainty head is {}, and sharpness of richards curve learnable parameter is {}.'.format(cfg.CUSTOM_OPTIONS.DETECTOR_TYPE, cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG, cfg.DATASETS.TRAIN[0], cfg.SOLVER.IMS_PER_BATCH,
				cfg.SOLVER.BASE_LR, cfg.MODEL.WEIGHTS, cfg.SOLVER.MAX_ITER, cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.MAX_ITER / 118287.0, cfg.SOLVER.CHECKPOINT_PERIOD, cfg.CUSTOM_OPTIONS.NEW_UNCERTAINTY_HEAD, cfg.CUSTOM_OPTIONS.LEARN_RC_SHARPNESS)
else:
	model_info = 'This is a {} model being trained with {} loss. We are training it on {} dataset with batchsize {} and intial learning rate of {}. This model intialzes weights from {} file. We are training this model for {} iterations, which roughly translates into {} epochs. Our stage-1 is {}. Everything else is unfrozen. We store checkpoints at every {} iterations. In this model, existence of complex uncertainty head is {}, and sharpness of richards curve learnable parameter is {}.'.format(cfg.CUSTOM_OPTIONS.DETECTOR_TYPE, cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG, cfg.DATASETS.TRAIN[0], cfg.SOLVER.IMS_PER_BATCH,
				cfg.SOLVER.BASE_LR, cfg.MODEL.WEIGHTS, cfg.SOLVER.MAX_ITER, cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.MAX_ITER / 118287.0, stage_1_info, cfg.SOLVER.CHECKPOINT_PERIOD, cfg.CUSTOM_OPTIONS.NEW_UNCERTAINTY_HEAD, cfg.CUSTOM_OPTIONS.LEARN_RC_SHARPNESS)

text_file = open(os.path.join(cfg.OUTPUT_DIR, 'model_info.txt'), "w")
text_file.write("model_info: %s" % model_info)
text_file.close()

print(model_info)

trainer.train()


