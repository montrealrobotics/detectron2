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


# ap = argparse.ArgumentParser()
# ap.add_argument("-name", "--experiment_comment", required = True, help="Comments for the experiment")

# args = vars(ap.parse_args())

# dir_name = args['experiment_comment']

class_list = ['Car', 'Van', 'Truck', 'Tram']

# write a function that loads the dataset into detectron2's standard format
def get_kitti_dicts(root_dir, data_label):
    
    image_names = sorted(glob.glob(root_dir+"/images/training/*.png"))
    train_images = int(len(image_names)*0.75)
    test_images = len(image_names) - train_images
    if data_label == 'train':
        image_names = image_names[:train_images]
        image_names = image_names[0:10]
    if data_label == 'test':
        # import ipdb; ipdb.set_trace()
        image_names = image_names[-test_images:]
    # print(image_names)
    # image_names = image_names[0:10]
        
    record = {}
    dataset_dicts = []

    
    class_label = {}
    ## giving labels to the classes
    for idx,class_val in enumerate(class_list):
        class_label[class_val] = idx

    for idx, name in enumerate(image_names):
        # print(name)
        record = {}
        height, width = cv2.imread(name).shape[:2]
        # import ipdb; ipdb.set_trace()
        record["file_name"] = name
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        label_name = root_dir+"/labels/training/"+name[-10:-3]+"txt"
        print(label_name)
        ob_list = []
        ## Creating a dictionary expected by detectron
        with open(label_name) as file:
            objects = file.read().splitlines()
            objs = []
            for obj in objects:
                obj = obj.split()
                if obj[0] in class_list:
                    obj_ann = {
                        "bbox": [float(i) for i in obj[4:8]],
                        "bbox_mode": BoxMode.XYXY_ABS,                    
                        "category_id": class_label[obj[0]],
                        "iscrowd": 0
                    }

                    objs.append(obj_ann)
            # print(len(objs))
            record["annotations"] = objs
            # print(record["annotations"])
        dataset_dicts.append(record)

        # if data_label == 'test':
            
        #     for name in image_names:
        #         # print(name)
        #         record = {}
        #         height, width = cv2.imread(name).shape[:2]
        #         record["file_name"] = name
        #         record["height"] = height
        #         record["width"] = width
        #         dataset_dicts.append(record)

    return dataset_dicts



from detectron2.data import DatasetCatalog, MetadataCatalog

root_dir = '/network/tmp1/bhattdha/kitti_dataset'
for d in ["train", "test"]:
    DatasetCatalog.register("kitti/" + d, lambda d=d: get_kitti_dicts(root_dir, d))
    MetadataCatalog.get('kitti/' + d).set(thing_classes=class_list)

kitti_metadata = MetadataCatalog.get('kitti/test')

# import ipdb; ipdb.set_trace()
print("data loading")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg1 = get_cfg()
cfg2 = get_cfg()
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_deform_conv_3x.yaml")

# loading config used during train time
cfg_dict1 = torch.load('/network/tmp1/bhattdha/detectron2_kitti/resnet-101_32x8d_FPN_deform_conv/resnet-101_32x8d_FPN_deform_conv_cfg.final')
cfg_dict2 = torch.load('/network/tmp1/bhattdha/detectron2_kitti/resnet-50_FPN/resnet-50_FPN_cfg.final')

cfg1 = cfg_dict1['cfg']
cfg2 = cfg_dict2['cfg']

cfg1.MODEL.RPN.POST_NMS_TOPK_TEST = 500
cfg2.MODEL.RPN.POST_NMS_TOPK_TEST = 500

cfg1.DATASETS.TEST = ("kitti/test",)   # no metrics implemented for this dataset
cfg1.DATALOADER.NUM_WORKERS = 2
cfg1.SOLVER.IMS_PER_BATCH = 6
cfg2.DATASETS.TEST = ("kitti/test",)   # no metrics implemented for this dataset
cfg2.DATALOADER.NUM_WORKERS = 2
cfg2.SOLVER.IMS_PER_BATCH = 6
# cfg.SOLVER.BASE_LR = 0.015
# cfg.SOLVER.BASE_LR = 3e-4  
# cfg.SOLVER.MAX_ITER =  200000  
cfg1.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg1.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)  #  (kitti)
cfg2.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg2.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)  #  (kitti)


cfg1.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_kitti/resnet-101_32x8d_FPN_deform_conv/'
cfg2.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_kitti/resnet-50_FPN/'
# import pdb; pdb.set_trace()

"""Now, we perform inference with the trained model on the kitti dataset. First, let's create a predictor using the model we just trained:"""
cfg1.MODEL.WEIGHTS = os.path.join(cfg1.OUTPUT_DIR, "model_0035999.pth")
cfg1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg2.MODEL.WEIGHTS = os.path.join(cfg2.OUTPUT_DIR, "model_0029999.pth")
cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_kitti/' + dir_name

if cfg1.CUSTOM_OPTIONS.DETECTOR_TYPE is 'deterministic':
    ## has to be smooth l1 loss if detector is deterministc
    cfg1.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'smooth_l1'

if cfg2.CUSTOM_OPTIONS.DETECTOR_TYPE is 'deterministic':
    ## has to be smooth l1 loss if detector is deterministc
    cfg2.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'smooth_l1'

predictor1 = DefaultPredictor(cfg1)
predictor2 = DefaultPredictor(cfg2)
# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=True)



# import ipdb; ipdb.set_trace()

## Evaluation happens here
from detectron2.evaluation import COCOEvaluator, inference_on_dataset_mm
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("kitti/test", cfg1, False, output_dir="./random/")
val_loader = build_detection_test_loader(cfg1, "kitti/test")
results  = inference_on_dataset_mm(predictor2.model, predictor2.model, val_loader, evaluator)
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
