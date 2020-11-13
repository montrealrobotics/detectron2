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
import argparse
from cityscapes_custom_load import load_cityscapes_train
from cityscapesscripts.helpers.labels import labels
from detectron2.data import DatasetCatalog, MetadataCatalog

ap = argparse.ArgumentParser()
ap.add_argument("-name", "--experiment_comment", required = True, help="Comments for the experiment")
args = vars(ap.parse_args())

dir_name = args['experiment_comment']

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_R_26_FPN_3x.yaml")
cfg.merge_from_file("/home/mila/b/bhattdha/detectron2/configs/Cityscapes/faster_rcnn_R_101_FPN_3x.yaml")



######################################################################################################################################################

# cfg.CUSTOM_OPTIONS.TEMP_SCALE_TRAINING = True
# cfg.CUSTOM_OPTIONS.TEMP_SCALE_ENABLED = True

if cfg.CUSTOM_OPTIONS.TEMP_SCALE_TRAINING:
	DatasetCatalog.register("cityscapes_temp_scaling", lambda d='train': load_cityscapes_train(temp_scaling = cfg.CUSTOM_OPTIONS.TEMP_SCALE_ENABLED))
	thing_classes = [k.name for k in labels if k.hasInstances and not k.ignoreInEval]
	MetadataCatalog.get("cityscapes_temp_scaling").set(thing_classes=thing_classes)
	cfg.DATASETS.TRAIN = ("cityscapes_temp_scaling",)

######################################################################################################################################################


# cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 8
# cfg.CUSTOM_OPTIONS.LOSS_WEIGHTS = [3.0, 1.0] 
cfg.MODEL.WEIGHTS = "/network/tmp1/bhattdha/detectron2_cityscapes/cityscapes_loss_attenuation/model_0009999.pth"

cfg.SOLVER.IMS_PER_BATCH = 15
# cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.BASE_LR = 1e-4
cfg.SOLVER.MAX_ITER =  15000
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_cityscapes/' + dir_name
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.CUSTOM_OPTIONS.DETECTOR_TYPE = 'probabilistic'
cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'loss_att'
cfg.CUSTOM_OPTIONS.ENCODING_TYPE = 'xyxy'

if cfg.CUSTOM_OPTIONS.DETECTOR_TYPE is 'deterministic':
    ## has to be smooth l1 loss if detector is deterministc
    cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'smooth_l1'


# cfg.CUSTOM_OPTIONS.LEARN_RC_SHARPNESS = True
# cfg.CUSTOM_OPTIONS.NEW_UNCERTAINTY_HEAD = True

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

### At this point, we will save the config as it becomes vital for testing in future
torch.save({'cfg': cfg}, cfg.OUTPUT_DIR + '/' + dir_name + '_cfg.final')
with PathManager.open(cfg.OUTPUT_DIR + '/' + dir_name + '_cfg.yaml', "w") as f:
	f.write(cfg.dump())

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)

uncertainty_heads_name = ['roi_heads.box_predictor.bbox_uncertainty_pred.weight',
'roi_heads.box_predictor.bbox_uncertainty_pred.bias']
regression_heads_name = ['roi_heads.box_predictor.bbox_pred.weight',
'roi_heads.box_predictor.bbox_pred.bias']

stage_1_info = 'unfrozen'
uncertainty_head = None

for name, p in trainer.model.named_parameters():

	## if we are learning temprature parameter, we
	## freeze entire model except temprature parameter
	
	if cfg.CUSTOM_OPTIONS.TEMP_SCALE_ENABLED:
		if name == 'roi_heads.box_predictor.temp_scale':
			p.requires_grad = True
			continue
		else:
			p.requires_grad = False
			continue
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
				cfg.SOLVER.BASE_LR, cfg.MODEL.WEIGHTS, cfg.SOLVER.MAX_ITER, cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.MAX_ITER / 2975.0, cfg.SOLVER.CHECKPOINT_PERIOD,  cfg.CUSTOM_OPTIONS.NEW_UNCERTAINTY_HEAD, cfg.CUSTOM_OPTIONS.LEARN_RC_SHARPNESS)
else:
	model_info = 'This is a {} model being trained with {} loss. We are training it on {} dataset with batchsize {} and intial learning rate of {}. This model intialzes weights from {} file. We are training this model for {} iterations, which roughly translates into {} epochs. Our stage-1 is {}. Everything else is unfrozen. We store checkpoints at every {} iterations. In this model, existence of complex uncertainty head is {}, and sharpness of richards curve learnable parameter is {}.'.format(cfg.CUSTOM_OPTIONS.DETECTOR_TYPE, cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG, cfg.DATASETS.TRAIN[0], cfg.SOLVER.IMS_PER_BATCH,
				cfg.SOLVER.BASE_LR, cfg.MODEL.WEIGHTS, cfg.SOLVER.MAX_ITER, cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.MAX_ITER / 2975.0, stage_1_info, cfg.SOLVER.CHECKPOINT_PERIOD,  cfg.CUSTOM_OPTIONS.NEW_UNCERTAINTY_HEAD, cfg.CUSTOM_OPTIONS.LEARN_RC_SHARPNESS)

if cfg.CUSTOM_OPTIONS.TEMP_SCALE_ENABLED:
	model_info = 'This is a {} model being trained with {} loss. We are training it on {} dataset with batchsize {} and intial learning rate of {}. This model intialzes weights from {} file. We are training this model for {} iterations, which roughly translates into {} epochs. We are learning temprature parameter in this model.'.format(cfg.CUSTOM_OPTIONS.DETECTOR_TYPE, cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG, cfg.DATASETS.TRAIN[0], cfg.SOLVER.IMS_PER_BATCH,
				cfg.SOLVER.BASE_LR, cfg.MODEL.WEIGHTS, cfg.SOLVER.MAX_ITER, cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.MAX_ITER / 2975.0)
text_file = open(os.path.join(cfg.OUTPUT_DIR, 'model_info.txt'), "w")
text_file.write("model_info: %s" % model_info)
text_file.close()

print(model_info)

print("Start training!")
print("The checkpoint iteration value is: ", cfg.SOLVER.CHECKPOINT_PERIOD)
trainer.train()
    