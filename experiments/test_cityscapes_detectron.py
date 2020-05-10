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
from detectron2.utils.visualizer import ColorMode

cfg = get_cfg()
cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/Cityscapes/faster_rcnn_R_101_FPN_3x.yaml")

cs_metadata = MetadataCatalog.get('cityscapes_fine_instance_seg_val')


"""Now, let's fine-tune a coco-pretrained R50-FPN Mask R-CNN model on the balloon dataset. It takes ~6 minutes to train 300 iterations on Colab's K80 GPU."""

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_R_152_FPN_3x.yaml")


# loading config used during train time
# cfg_dict = torch.load('/network/tmp1/bhattdha/detectron2_kitti/resnet-50_FPN/resnet-50_FPN_cfg.final')
cfg = torch.load('/network/tmp1/bhattdha/detectron2_cityscapes/resnet-101_cityscapes/resnet-101_cityscapes_cfg.final')['cfg']

# cfg.DATASETS.TRAIN = ("kitti/train",)
# cfg.DATASETS.TEST = ('coco_2017_val',)   # no metrics implemented for this dataset

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset

cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_cityscapes/resnet-101_cityscapes/'

"""Now, we perform inference with the trained model on the kitti dataset. First, let's create a predictor using the model we just trained:"""
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0249999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

"""Then, we randomly select several samples to visualize the prediction results."""
time_inference = []
image_names = glob.glob("/network/tmp1/bhattdha/cityscapes/leftImg8bit/test/*/*.png")
for idx, im_name in enumerate(image_names):   
    print(idx, im_name)
    im = cv2.imread(im_name)
    import time
    st_time = time.time()
    outputs = predictor(im)
    spred_boxes = outputs['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy()
    pred_classes = outputs['instances'].get_fields()['pred_classes'].cpu().numpy()
    tot_time = time.time() - st_time
    print("total time per image is:", tot_time)
    time_inference.append(tot_time)
    # import pdb; pdb.set_trace()
    # pdb.set_trace()
    v = Visualizer(im[:, :, ::-1],
                   metadata=cs_metadata, 
                   scale=1.0, 
                   instance_mode=ColorMode.IMAGE   
    )

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # print("saving images")
    # print(type(v))
    cv2.imwrite('/network/tmp1/bhattdha/detectron2_cityscapes/resnet-101_cityscapes/outputs/' + str(idx).zfill(6) + '.png', v.get_image()[:, :, ::-1]) 
    # cv2_imshow(v.get_image()[:, :, ::-1])
    
print("Average time is:", np.mean(np.array(time_inference)))


# kitti_tracking_dict = '/network/tmp1/bhattdha/kitti_tracking/'
# kitti_seqs = glob.glob(kitti_tracking_dict + '*')
# kitti_seqs.sort()
# final_result = {}

# for seq in kitti_seqs:
#     print(seq)
#     images_dict = {}
#     images_dict_list = []
#     print(seq[-4:])
#     images = glob.glob(seq + '/*.png')
#     images.sort()
#     for image in images:
        
#         print(image[-10:])
#         im = cv2.imread(image)
#         outputs = predictor(im)
#         pred_boxes = outputs['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy()
#         pred_classes = outputs['instances'].get_fields()['pred_classes'].cpu().numpy()
#         ## for car class
#         ind = np.where(pred_classes==0)[0]
#         # if (pred_classes!=0).sum() > 0:
#         #     import ipdb; ipdb.set_trace()
#         pred_boxes = pred_boxes[ind,:]
#         images_dict[image[-10:]] = pred_boxes
#         images_dict_list.append(images_dict)

#     final_result[seq[-4:]] = images_dict_list


import ipdb; ipdb.set_trace()