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

class_list = ['Car', 'Van', 'Truck', 'Tram']

# write a function that loads the dataset into detectron2's standard format
def get_kitti_dicts(root_dir, data_label):
    
    
    if data_label == 'train':
        image_names = glob.glob(root_dir+"/images/training/*.png")
    if data_label == 'test':
        image_names = glob.glob(root_dir+"/images/testing/*.png")
    # print(image_names)
    image_names = image_names[0:20]
    print("total images are:", image_names)
    record = {}
    dataset_dicts = []

    
    class_label = {}
    ## giving labels to the classes
    for idx,class_val in enumerate(class_list):
        class_label[class_val] = idx

    for idx, name in enumerate(image_names):
        print(idx, name)
        # print(name)
        record = {}
        height, width = cv2.imread(name).shape[:2]
        record["file_name"] = name
        record["height"] = height
        record["width"] = width
        if data_label == 'train':
            label_name = root_dir+"/labels/training/"+name[-10:-3]+"txt"
                # print(label_name)
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


"""Now, let's fine-tune a coco-pretrained R50-FPN Mask R-CNN model on the balloon dataset. It takes ~6 minutes to train 300 iterations on Colab's K80 GPU."""

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_R_152_FPN_3x.yaml")


# loading config used during train time
# cfg_dict = torch.load('/network/tmp1/bhattdha/detectron2_kitti/resnet-50_FPN/resnet-50_FPN_cfg.final')
cfg_dict = torch.load('/network/tmp1/bhattdha/detectron2_kitti/resnet-101_32x8d_FPN_deform_conv/resnet-101_32x8d_FPN_deform_conv_cfg.final')
cfg = cfg_dict['cfg']

# cfg.DATASETS.TRAIN = ("kitti/train",)
cfg.DATASETS.TEST = ('coco_2017_val',)   # no metrics implemented for this dataset

# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"  # initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 12
# cfg.SOLVER.BASE_LR = 0.015  
# cfg.SOLVER.MAX_ITER =  40000   # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)  #  (kitti)


# import pdb; pdb.set_trace()
cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_kitti/resnet-101_32x8d_FPN_deform_conv/'
# import pdb; pdb.set_trace()

"""Now, we perform inference with the trained model on the kitti dataset. First, let's create a predictor using the model we just trained:"""
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0039999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# cfg.DATASETS.TEST = ("kitti/test", )
predictor = DefaultPredictor(cfg)

"""Then, we randomly select several samples to visualize the prediction results."""



from detectron2.utils.visualizer import ColorMode
time_inference = []


im = cv2.imread('/network/home/bhattdha/detectron2/experiments/messedup.jpg')
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
               metadata=kitti_metadata, 
               scale=1.0, 
               instance_mode=ColorMode.IMAGE   
)

v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# print("saving images")
    # print(type(v))
cv2.imwrite('messedup_out.png', v.get_image()[:, :, ::-1]) 
import ipdb; ipdb.set_trace()
# dataset_dicts = get_kitti_dicts("/network/tmp1/bhattdha/kitti_dataset", 'test')
# image_names = glob.glob(root_dir+"/images/testing/*.png")
# for idx, im_name in enumerate(image_names):   
#     print(idx, im_name)
    # im = cv2.imread(im_name)
    # import time
    # st_time = time.time()
    # outputs = predictor(im)
    # spred_boxes = outputs['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy()
    # pred_classes = outputs['instances'].get_fields()['pred_classes'].cpu().numpy()
#     import ipdb; ipdb.set_trace()
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
#     cv2.imwrite("/network/tmp1/bhattdha/detectron2_kitti/resnet-50_FPN/test_outputs/" + str(idx).zfill(6) + '.png', v.get_image()[:, :, ::-1]) 
#     # cv2_imshow(v.get_image()[:, :, ::-1])
    
# print("Average time is:", np.mean(np.array(time_inference)))


kitti_tracking_dict = '/network/tmp1/bhattdha/kitti_tracking/'
kitti_seqs = glob.glob(kitti_tracking_dict + '*')
kitti_seqs.sort()
final_result = {}

for seq in kitti_seqs:
    print(seq)
    images_dict = {}
    images_dict_list = []
    print(seq[-4:])
    images = glob.glob(seq + '/*.png')
    images.sort()
    for image in images:
        
        print(image[-10:])
        im = cv2.imread(image)
        outputs = predictor(im)
        pred_boxes = outputs['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy()
        pred_classes = outputs['instances'].get_fields()['pred_classes'].cpu().numpy()
        ## for car class
        ind = np.where(pred_classes==0)[0]
        # if (pred_classes!=0).sum() > 0:
        #     import ipdb; ipdb.set_trace()
        pred_boxes = pred_boxes[ind,:]
        images_dict[image[-10:]] = pred_boxes
        images_dict_list.append(images_dict)

    final_result[seq[-4:]] = images_dict_list

import ipdb; ipdb.set_trace()