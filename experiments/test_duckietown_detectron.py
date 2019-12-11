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
import pdb
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
class_list = ['cone','duckie','duckiebot']


"""Now, let's fine-tune a coco-pretrained R50-FPN Mask R-CNN model on the balloon dataset. It takes ~6 minutes to train 300 iterations on Colab's K80 GPU."""

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()

cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

class_list = ['cone','duckie','duckiebot']

# write a function that loads the dataset into detectron2's standard format
def get_duckietown_dicts(root_dir):
    
    
    annotation_file = root_dir + 'annotations/final_anns.json'
    frame_path = root_dir + 'final_frames/frames/'

    with open(annotation_file) as f: 
        data = json.load(f) 

    record = {}
    dataset_dicts = []

    
    class_label = {}
    ## giving labels to the classes
    for idx,class_val in enumerate(class_list):
        class_label[class_val] = idx 

    for name in data.keys():
        # print(name)
        image_name = frame_path + name
        record = {}
        height, width = cv2.imread(image_name).shape[:2]

        record["file_name"] = image_name
        record["height"] = height
        record["width"] = width
        
        objs = []

        for annotation in data[name]:
            
            ob_list = []
            obj_ann = {
                            "bbox": [annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][0] + annotation['bbox'][2], annotation['bbox'][1] + annotation['bbox'][3]],
                            "bbox_mode": BoxMode.XYXY_ABS,                    
                            "category_id": annotation['cat_id'] - 1,
                            "iscrowd": 0
                        }
            objs.append(obj_ann)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


from detectron2.data import DatasetCatalog, MetadataCatalog

root_dir = '/network/tmp1/bhattdha/duckietown_dataset/'

for d in ["train", "test"]:
    DatasetCatalog.register("duckietown/" + d, lambda d=d: get_duckietown_dicts(root_dir))
    MetadataCatalog.get('duckietown/' + d).set(thing_classes=class_list)

duckietown_metadata = MetadataCatalog.get('duckietown/train')

cfg_load = torch.load('/network/tmp1/bhattdha/duckietown_dataset/probabilistic_duckietown_OD/probabilistic_duckietown_OD_cfg.final')

##loading the config used at train time
cfg = cfg_load['cfg']
# import pdb; pdb.set_trace()
# cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATASETS.TEST = ('coco_2017_val',)   # no metrics implemented for this dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)  #  (kitti)
cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/duckietown_dataset/probabilistic_duckietown_OD/'
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)  #  (kitti)

"""Now, we perform inference with the trained model on the kitti dataset. First, let's create a predictor using the model we just trained:"""
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0014999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model


# cfg.DATASETS.TEST = ("kitti/test", )
predictor = DefaultPredictor(cfg)

"""Then, we randomly select several samples to visualize the prediction results."""

from detectron2.utils.visualizer import ColorMode

# im = cv2.imread('test.png')
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1],
#                    metadata=duckietown_metadata, 
#                    scale=1.0, 
#                    instance_mode=ColorMode.IMAGE   
#     )

# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# cv2.imwrite("test_out.png", v.get_image()[:, :, ::-1]) 

# import pdb; pdb.set_trace()


# import time
# inf_time = []
# # If the input is the camera, pass 0 instead of the video file name
# cap = cv2.VideoCapture('/network/home/bhattdha/manfred_vid.mov')
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# out = cv2.VideoWriter('/network/home/bhattdha/output_prob.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
# while(cap.isOpened()):

    
#     ret, frame = cap.read()
#     st_time = time.time()
#     outputs = predictor(frame)
#     end_time = time.time() - st_time
#     inf_time.append(time.time() - st_time)
#     # pdb.set_trace()
#     v = Visualizer(frame[:, :, ::-1],
#                    metadata=duckietown_metadata, 
#                    scale=1.0, 
#                    instance_mode=ColorMode.IMAGE   
#     )
#     # out.write(frame)
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     print("Tot time is: ", end_time)
#     # print(type(v))
#     # import ipdb; ipdb.set_trace()
#     out.write(v.get_image()[:, :, ::-1])

# # When everything done, release the video capture and video write objects
# cap.release()
# out.release()
# print("Inference time: ", np.mean(np.array(inf_time)))

# dataset_dicts = get_kitti_dicts("/network/tmp1/bhattdha/kitti_dataset", 'test')
image_names = glob.glob("/network/tmp1/bhattdha/duckietown_dataset/final_frames/test/*.png")

for idx, im_name in enumerate(image_names):   
    print(idx, im_name)
    im = cv2.imread(im_name)
    outputs = predictor(im)
    # pdb.set_trace()
    v = Visualizer(im[:, :, ::-1],
                   metadata=duckietown_metadata, 
                   scale=1.0, 
                   instance_mode=ColorMode.IMAGE   
    )

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    print("saving images")
    print(type(v))
    cv2.imwrite("/network/tmp1/bhattdha/duckietown_dataset/probabilistic_duckietown_OD/test_outputs/" + str(idx).zfill(5) + '.png', v.get_image()[:, :, ::-1]) 
    # cv2_imshow(v.get_image()[:, :, ::-1])
    