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
import argparse
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import pathlib
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

import argparse

# os.environ["DETECTRON2_DAT"] = "my_export"

ap = argparse.ArgumentParser()
ap.add_argument("-name", "--model_directory", required = True, help="Directory where model is located") ##  example: coco_xyxy_full_model

## example: model_0004999.pth
## if model_name is "all", then we evaluate all the models in the directory
ap.add_argument("-model_name", "--model_to_evaluate", required = True, help="Which model to evaluate")  

args = vars(ap.parse_args())

dir_name = args['model_directory']
model_name = args['model_to_evaluate']

model_dir_path = os.path.join('/network/tmp1/bhattdha/detectron2_coco/', dir_name)
if model_name == "all":
    model_paths = sorted(glob.glob(model_dir_path + '/model_*.pth'))
else:
    model_paths = sorted(glob.glob(model_dir_path + '/' + model_name), reverse=True)

## if we have no models, we can't evaluate them
assert len(model_paths) != 0, 'No models found in {} directory'.format(model_dir_path)

# model_paths = sorted(glob.glob('/network/tmp1/bhattdha/detectron2_coco/coco_xyxy_full_model/model_*.pth'), reverse=True)
# model_paths = ["/home/mila/b/bhattdha/model_final_a3ec72.pkl"]

final_results = {}

for model_path in model_paths:
# import ipdb; ipdb.set_trace()
    
    model_name = os.path.basename(model_path)
    print(model_path)
    cfg = get_cfg()
    # cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    # cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_deform_conv_3x.yaml")

    # loading config used during train time
    # cfg_dict = torch.load('/network/tmp1/bhattdha/detectron2_kitti/resnet-50_FPN/resnet-50_FPN_cfg.final')
    cfg.merge_from_file('/network/tmp1/bhattdha/detectron2_coco/' + dir_name + '/' + dir_name + '_cfg.yaml')

    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    
    cfg.DATASETS.TEST = ("coco_2017_val",)
    cfg.MODEL.MASK_ON = False
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 10
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    ### test isotonic regression model
    cfg.CUSTOM_OPTIONS.ISOTONIC_REG = True
    cfg.CUSTOM_OPTIONS.ISOTONIC_MODEL_PATH = '/home/mila/b/bhattdha/detectron2/detectron2/modeling/roi_heads/isotonic_reg_gp_beta/models/coco_isotonic_loss_att.pkl'

    cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_coco/' + dir_name + '/'
    # cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_kitti/resnet-50_FPN/'
    # import pdb; pdb.set_trace()

    """Now, we perform inference with the trained model on the kitti dataset. First, let's create a predictor using the model we just trained:"""
    cfg.MODEL.WEIGHTS = model_path
    # cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_kitti/' + dir_name

    predictor = DefaultPredictor(cfg)
    # trainer = DefaultTrainer(cfg) 
    # trainer.resume_or_load(resume=True)

    # params_count = sum(p.numel() for p in predictor.model.parameters())
    # print('total number of parameters: {}'.format(params_count))
    # import ipdb; ipdb.set_trace()
    ## Evaluation happens here
    
    evaluator = COCOEvaluator("coco_2017_val", cfg, False, output_dir=os.path.join(model_dir_path, 'results', model_name[:-4]))
    val_loader = build_detection_test_loader(cfg, "coco_2017_val")
    results  = inference_on_dataset(predictor.model, val_loader, evaluator)
    final_results[model_name] = results
    np.save(os.path.join(model_dir_path, 'results', model_name[:-4], 'results.npy'), results)
    # print(results)

if model_name == "all":
    np.save(os.path.join(model_dir_path, 'results','final_results.npy'), final_results)

# from detectron2.utils.visualizer import ColorMode
# time_inference = []
# # dataset_dicts = get_kitti_dicts("/network/tmp1/bhattdha/kitti_dataset", 'test')
# image_names = glob.glob(root_dir+"/images/testing/*.png")
# for idx, im_name in enumerate(image_names):   
#     print(idx, im_name)
#     im = cv2.imread(im_name)
#     import time
#     st_time = time.time()
#     outputs = predictor(im)
#     tot_time = time.time() - st_time
#     print("total time per image is:", tot_time)
#     time_inference.append(tot_time)
#     # import pdb; pdb.set_trace()
#     # pdb.set_trace()
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=kitti_metadata, 
#                    scale=1.0, 
#                    instance_mode=ColorMode.IMAGE   
#     )

#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     # print("saving images")
#     # print(type(v))
#     cv2.imwrite("/network/tmp1/bhattdha/detectron2_kitti/resnet-50_FPN/test_outputs2/" + str(idx).zfill(6) + '.png', v.get_image()[:, :, ::-1]) 
