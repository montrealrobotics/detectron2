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


ap = argparse.ArgumentParser()
ap.add_argument("-name", "--experiment_comment", required = True, help="Comments for the experiment")

args = vars(ap.parse_args())

dir_name = args['experiment_comment']

# class_list = ['Car', 'Van', 'Truck', 'Tram']
class_list = ['Pedestrian', 'Cyclist', 'Person_sitting', 'Tram']


# write a function that loads the dataset into detectron2's standard format
def get_kitti_dicts(root_dir, data_label):
    
    image_names = sorted(glob.glob(root_dir+"/images/training/*.png"))
    train_images = int(len(image_names)*0.75)
    test_images = len(image_names) - train_images
    if data_label == 'train':
        image_names = image_names[:train_images]
        image_names = image_names[0:10]
    if data_label == 'test':
        image_names = image_names[-test_images:]
    # print(image_names)
    # image_names = image_names[0:10]
    # import ipdb; ipdb.set_trace()
    record = {}
    dataset_dicts = []

    
    class_label = {}
    ## giving labels to the classes
    for idx,class_val in enumerate(class_list):
        class_label[class_val] = idx

    for name in image_names:
        # print(name)
        record = {}
        height, width = cv2.imread(name).shape[:2]
        record["file_name"] = name
        record["height"] = height
        record["width"] = width
        if data_label == 'train':
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

        if data_label == 'test':
            
            for name in image_names:
                # print(name)
                record = {}
                height, width = cv2.imread(name).shape[:2]
                record["file_name"] = name
                record["height"] = height
                record["width"] = width
                dataset_dicts.append(record)

    return dataset_dicts



from detectron2.data import DatasetCatalog, MetadataCatalog

root_dir = '/network/tmp1/bhattdha/kitti_dataset'
for d in ["train", "test"]:
    DatasetCatalog.register("kitti/" + d, lambda d=d: get_kitti_dicts(root_dir, d))
    MetadataCatalog.get('kitti/' + d).set(thing_classes=class_list)

kitti_metadata = MetadataCatalog.get('kitti/train')


print("data loading")
# dataset_dicts = get_kitti_dicts(root_dir, 'train')
# print(dataset_dicts[0]['annotations'])

# for d in random.sample(dataset_dicts, 15):
#     # print(d)
#     # print(d["file_name"])
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_metadata, scale=1.0)
#     vis = visualizer.draw_dataset_dict(d)
    # print(vis.get_image()[:, :, ::-1].shape)   
    # while(1):
    #     cv2.imshow('', vis.get_image()[:, :, ::-1])
    #     k = cv2.waitKey(33)
    #     if k == -1:
    #         continue
    #     elif k == 27:
    #         break



"""Now, let's fine-tune a coco-pretrained R50-FPN Mask R-CNN model on the balloon dataset. It takes ~6 minutes to train 300 iterations on Colab's K80 GPU."""

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_R_26_FPN_3x.yaml")
cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_deform_conv_3x.yaml")
cfg.DATASETS.TRAIN = ("kitti/train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # initialize from model zoo
# cfg.MODEL.WEIGHTS = "/network/tmp1/bhattdha/detectron2_kitti/resnet-26_FPN_3x_scratch/model_start.pth"
# cfg.MODEL.WEIGHTS = "/network/tmp1/bhattdha/detectron2_kitti/model_0014999.pth"  # initialize fron deterministic model
cfg.SOLVER.IMS_PER_BATCH = 1
# cfg.SOLVER.BASE_LR = 0.015
cfg.SOLVER.BASE_LR = 2e-4  
cfg.SOLVER.MAX_ITER =  250000  
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)  #  (kitti)
cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_kitti/' + dir_name

cfg.MODEL.RPN.IOU_THRESHOLDS = [0.00005, 0.5]
cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.00005, 0.5]
cfg.MODEL.ROI_HEADS.IOU_LABELS = [0, -1, 1]
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.99
cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = False

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


# """Now, we perform inference with the trained model on the balloon validation dataset. First, let's create a predictor using the model we just trained:"""

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# cfg.DATASETS.TEST = ("kitti/test", )
# predictor = DefaultPredictor(cfg)

# """Then, we randomly select several samples to visualize the prediction results."""

# from detectron2.utils.visualizer import ColorMode
# dataset_dicts = get_kitti_dicts("/network/tmp1/bhattdha/kitti_dataset", 'test')
# for d in random.sample(dataset_dicts, 3):    
#     im = cv2.imread(d["file_name"])
#     print(im)
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=kitti_metadata, 
#                    scale=1.0, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#     )

#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
#     # cv2_imshow(v.get_image()[:, :, ::-1])
    