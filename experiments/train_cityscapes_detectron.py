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


ap = argparse.ArgumentParser()
ap.add_argument("-name", "--experiment_comment", required = True, help="Comments for the experiment")
args = vars(ap.parse_args())

dir_name = args['experiment_comment']

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_R_26_FPN_3x.yaml")
cfg.merge_from_file("/home/mila/b/bhattdha/detectron2/configs/Cityscapes/faster_rcnn_R_101_FPN_3x.yaml")

##########################################################################################################################
# cfg_dict = torch.load('/miniscratch/bhattdha/cityscapes_model/mask_rcnn_mask_on_finetuned_loss_att/mask_rcnn_mask_on_finetuned_loss_att_cfg.final')
# cfg = cfg_dict['cfg']
# cfg.MODEL.MASK_ON = False
##########################################################################################################################33


# cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 8
cfg.CUSTOM_OPTIONS.LOSS_WEIGHTS = [0.5, 0.5] 
cfg.MODEL.WEIGHTS = "/network/tmp1/bhattdha/detectron2_cityscapes/variance_loss_unfrozen_uncertainty_head/model_0001999.pth"
# cfg.MODEL.WEIGHTS = "/network/tmp1/bhattdha/detectron2_kitti/resnet-26_FPN_3x_scratch/model_start.pth"
# cfg.MODEL.WEIGHTS = "/network/tmp1/bhattdha/detectron2_kitti/model_0014999.pth"  # initialize fron deterministic model
cfg.SOLVER.IMS_PER_BATCH = 16
# cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.BASE_LR = 1e-3
cfg.SOLVER.MAX_ITER =  10000
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_cityscapes/' + dir_name
cfg.SOLVER.CHECKPOINT_PERIOD = 500
# cfg.MODEL.RPN.IOU_THRESHOLDS = [0.00005, 0.5]
# cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.00005, 0.5]
# cfg.MODEL.ROI_HEADS.IOU_LABELS = [0, -1, 1]
# cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.99
# cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
cfg.CUSTOM_OPTIONS.DETECTOR_TYPE = 'probabilistic'
cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'kl_div_chi_sq_closed_form_plus_smoothl1'

# cfg.CUSTOM_OPTIONS.DETECTOR_TYPE = 'deterministic'

if cfg.CUSTOM_OPTIONS.DETECTOR_TYPE is 'deterministic':
    ## has to be smooth l1 loss if detector is deterministc
    cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'smooth_l1'


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)






### At this point, we will save the config as it becomes vital for testing in future
torch.save({'cfg': cfg}, cfg.OUTPUT_DIR + '/' + dir_name + '_cfg.final')
with PathManager.open(cfg.OUTPUT_DIR + '/' + dir_name + '_cfg.yaml', "w") as f:
	f.write(cfg.dump())

#############################################################################################################################################

####################################### only use this code when you want to collect residuals################################################

# cfg.SOLVER.IMS_PER_BATCH = 10
# cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'collect_residuals'
# cfg.DATASETS.TRAIN = ("cityscapes_fine_instance_seg_val",)
# cfg.SOLVER.BASE_LR = 0
#############################################################################################################################################

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)

uncertainty_heads_name = ['roi_heads.box_predictor.bbox_uncertainty_pred.weight',
'roi_heads.box_predictor.bbox_uncertainty_pred.bias']
regression_heads_name = ['roi_heads.box_predictor.bbox_pred.weight',
'roi_heads.box_predictor.bbox_pred.bias']

for name, p in trainer.model.named_parameters():
	if 'roi_heads' not in name:
		print(name)
		p.requires_grad = False
	# if name not in uncertainty_heads_name: # and name not in regression_heads_name:
	# 	print(name)
	# 	p.requires_grad = False


print("Start training!")
print("The checkpoint iteration value is: ", cfg.SOLVER.CHECKPOINT_PERIOD)
trainer.train()
    