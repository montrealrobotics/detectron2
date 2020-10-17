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
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

model_paths = sorted(glob.glob('/network/tmp1/bhattdha/detectron2_cityscapes/first_stage_coco_frozen_R-101/model_0019999.pth'), reverse=True)
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
    cfg_dict = torch.load('/network/tmp1/bhattdha/detectron2_cityscapes/first_stage_coco_frozen_R-101/first_stage_coco_frozen_R-101_cfg.final')

    cfg = cfg_dict['cfg']
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    
    cfg.DATASETS.TEST = ("cityscapes_fine_instance_seg_val",)   
    cfg.MODEL.MASK_ON = False
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 10
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2

    cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_cityscapes/first_stage_coco_frozen_R-101/'
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
    
    evaluator = COCOEvaluator("cityscapes_fine_instance_seg_val", cfg, False, output_dir='first_stage_coco_frozen_R-101' + '_' + model_name)
    val_loader = build_detection_test_loader(cfg, "cityscapes_fine_instance_seg_val")
    results  = inference_on_dataset(predictor.model, val_loader, evaluator)
    final_results[model_name] = results
    # print(results)
print(final_results)
np.save(os.path.join(os.getcwd(), 'first_stage_coco_frozen_R-101', 'final_results.npy'), final_results)
import ipdb; ipdb.set_trace()

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
