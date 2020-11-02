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

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


ap = argparse.ArgumentParser()
ap.add_argument("-name", "--model_directory", required = True, help="Directory where model is located") ##  example: coco_xyxy_full_model

## example: model_0004999.pth
## if model_name is "all", then we evaluate all the models in the directory
ap.add_argument("-model_name", "--model_to_evaluate", required = True, help="Which model to evaluate")  

args = vars(ap.parse_args())

dir_name = args['model_directory']
model_name = args['model_to_evaluate']



model_dir_path = os.path.join('/network/tmp1/bhattdha/detectron2_coco/', dir_name)

model_full_path = os.path.join(model_dir_path, model_name)

assert os.path.exists(model_full_path), 'Given model named {} doesnt exist at path {}'.format(model_namem, model_dir_path)

cfg = get_cfg()
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_R_26_FPN_3x.yaml")
cfg.merge_from_file(os.path.join(model_dir_path, dir_name + '_cfg.yaml'))

# cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = model_full_path
# cfg.MODEL.WEIGHTS = "/network/tmp1/bhattdha/detectron2_kitti/resnet-26_FPN_3x_scratch/model_start.pth"
# cfg.MODEL.WEIGHTS = "/network/tmp1/bhattdha/detectron2_kitti/model_0014999.pth"  # initialize fron deterministic model
cfg.SOLVER.IMS_PER_BATCH = 10
cfg.CUSTOM_OPTIONS.RESIDUAL_MAX_ITER = 500
# cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.BASE_LR = 0
cfg.SOLVER.MAX_ITER =  1000000
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.OUTPUT_DIR = model_dir_path
cfg.SOLVER.CHECKPOINT_PERIOD = 10000

cfg.CUSTOM_OPTIONS.DETECTOR_TYPE = 'probabilistic'
cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'collect_residuals'
cfg.DATASETS.TRAIN = ("coco_2017_val",)

## filename by which the model's residuals is to be stored!
cfg.CUSTOM_OPTIONS.RESIDUAL_DIR_NAME = os.path.join(model_dir_path, 'residuals_storage')
cfg.CUSTOM_OPTIONS.MODEL_NAME = model_name[:-4]  ## we don't want ".pth"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.CUSTOM_OPTIONS.RESIDUAL_DIR_NAME, exist_ok=True)

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False) ## so it starts from the model we give

print("Start training!")
print("The checkpoint iteration value is: ", cfg.SOLVER.CHECKPOINT_PERIOD)
trainer.train()
    