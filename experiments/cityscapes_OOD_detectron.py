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
from detectron2.structures import ImageList
from detectron2.modeling import detector_postprocess
import copy
import torchvision
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.utils.visualizer import ColorMode

# ap = argparse.ArgumentParser()
# ap.add_argument("-name", "--experiment_comment", required = True, help="Comments for the experiment")

# args = vars(ap.parse_args())

# dir_name = args['experiment_comment']


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import glob

cfg = get_cfg()
cfg_rpn = get_cfg()
cfg_rpn.merge_from_file("/home/mila/b/bhattdha/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg_rpn.MODEL.WEIGHTS = "/home/mila/b/bhattdha/model_final_2d9806.pkl"
cfg_rpn.MODEL.META_ARCHITECTURE = "ProposalNetwork"
# cfg.merge_from_file("/network/home/bhattdha/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_deform_conv_3x.yaml")

# loading config used during train time
# cfg_dict = torch.load('/network/tmp1/bhattdha/detectron2_kitti/resnet-50_FPN/resnet-50_FPN_cfg.final')
cfg_dict = torch.load('/network/tmp1/bhattdha/detectron2_cityscapes/resnet-101_cityscapes/resnet-101_cityscapes_cfg.final')

cfg = cfg_dict['cfg']
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
# cfg.INPUT.MIN_SIZE_TEST = 400
# cfg.INPUT.MAX_SIZE_TEST = 800

cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_cityscapes/resnet-101_cityscapes/'
# cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_kitti/resnet-50_FPN/'
# import pdb; pdb.set_trace()

"""Now, we perform inference with the trained model on the kitti dataset. First, let's create a predictor using the model we just trained:"""
# cfg.MODEL.META_ARCHITECTURE = "ProposalNetwork"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0249999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# cfg.OUTPUT_DIR = '/network/tmp1/bhattdha/detectron2_kitti/' + dir_name

if cfg.CUSTOM_OPTIONS.DETECTOR_TYPE is 'deterministic':
    ## has to be smooth l1 loss if detector is deterministc
    cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'smooth_l1'

thresh_dict = np.load('/network/home/bhattdha/detectron2/generative_classifier/maha_thresh_cityscapes.npy',allow_pickle = True)[()]
means = thresh_dict['gaussian_stats']['means']
covars = thresh_dict['gaussian_stats']['covars']
maha_dist_thresh = thresh_dict['maha_thresh'][90] ## threshold for 95% accuracy on validation accuracy
predictor = DefaultPredictor(cfg)
rpn_pred = DefaultPredictor(cfg_rpn)
from detectron2.utils.visualizer import ColorMode

all_image_paths = glob.glob('/network/home/bhattdha/tensorflow_datasets/downloads/extracted/ZIP.dhbw-stuttgar.de_sgehrig_lostAndF_leftImg8MH9mACAjq1l9MJljuUmQ9bmo5XNe5ynDKSZHpm6fKxg.zip/leftImg8bit/train/*/*.png')
im = cv2.imread('/network/home/bhattdha/tensorflow_datasets/downloads/extracted/ZIP.dhbw-stuttgar.de_sgehrig_lostAndF_leftImg8MH9mACAjq1l9MJljuUmQ9bmo5XNe5ynDKSZHpm6fKxg.zip/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000012_000280_leftImg8bit.png')

## Region proposals can either come from originally trained model or via other pretrained models like Coco
rpn_out = ["original", "coco-RPN"]

def filter_wrong_oods(detected_objects, ood_outputs):

    '''
    detected_objects:  Nx4 output detections
    ood_outputs: Mx4 ood outputs
    '''

    M = ood_outputs.shape[0]
    N = detected_objects.shape[0]

    if N == 0:
        return ood_outputs

    true_ood_indices = []

    for i in range(M):

        keep_ood = True
        interarea_sum = 0

        for j in range(N):

            x_ood1, y_ood1, x_ood2, y_ood2 = ood_outputs[i]
            x_d1, y_d1, x_d2, y_d2 = detected_objects[j]

            xA = max(x_ood1, x_d1)
            yA = max(y_ood1, y_d1)
            xB = min(x_ood2, x_d2)
            yB = min(y_ood2, y_d2)
            # compute the area of intersection rectangle
            interArea = max(0, xB - xA) * max(0, yB - yA)
            ood_area = abs((x_ood2 - x_ood1)*(y_ood2 - y_ood1))

            if interArea/ood_area > 0.6:
                keep_ood = False


            interarea_sum += interArea

        if interarea_sum/ood_area > 0.8:
            keep_ood = False

        if keep_ood:
            true_ood_indices.append(i)

    return ood_outputs[true_ood_indices,:]

for ind, image_path in enumerate(all_image_paths):
    print("On image: ", image_path)
    im = cv2.imread(image_path)
    ## here comes the OOD detection(finger crossed)
    ## full model execution
    final_output = predictor(im)

    """
    let's get the region proposals from coco-RPN
    """
    coco_rpn_out = rpn_pred(im)
    coco_prop_rpn, coco_prop_rpn_processed = coco_rpn_out['proposals'], coco_rpn_out['processed_results']

    """
    let's partially execute the model
    """
    ### 1. preparing the input ###
    original_image = im
    if predictor.input_format == "RGB":
        # whether the model expects BGR inputs or RGB
        original_image = original_image[:, :, ::-1]
    height, width = original_image.shape[:2]

    ## we are keeping the original size of the image to see if there is any detection
    # image = original_image
    image = predictor.transform_gen.get_transform(original_image).apply_image(original_image)


    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]
    ### 2. forward pass through the backbone! ###
    assert not predictor.model.training
    batched_inputs = inputs
    images = predictor.model.preprocess_image(batched_inputs)
    features = predictor.model.backbone(images.tensor)

    ### 3. getting the region proposals ###
    proposals, _ = predictor.model.proposal_generator(images, features, None)
    proposals_process = copy.deepcopy(proposals)

    ### 4. processing region proposals ###
    processed_results = []
    for results_per_image, input_per_image, image_size in zip(
        proposals_process, batched_inputs, images.image_sizes
    ):
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])
        r = detector_postprocess(results_per_image, height, width)
        processed_results.append({"proposals": r})

    ### 5. Going through RoI heads, imp stuff coming in! ###
    features_list = [features[f] for f in predictor.model.roi_heads.in_features]
    box_features = predictor.model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in coco_prop_rpn])
    box_features = predictor.model.roi_heads.box_head(box_features)

    ### 6. Finding OODs now that we have embeddings ###
    ood_indices = []
    box_features_numpy = box_features.detach().clone().cpu().numpy()
    
    for i in range(len(box_features_numpy)):
        data_point = 1e2*box_features_numpy[i,:]
        ood_flag = True
        for class_label in maha_dist_thresh.keys():
            diff = (data_point - means[class_label]).reshape(len(data_point), 1)
            maha_dist = np.dot(diff.T, np.dot(covars[class_label], diff))
            if maha_dist < maha_dist_thresh[class_label]:
                ood_flag = False
                break
        if ood_flag:
            ood_indices.append(i)

    ### 7. Plotting detected objects
    cs_metadata = MetadataCatalog.get('cityscapes_fine_instance_seg_val')
    v = Visualizer(im[:, :, ::-1],
                   metadata=cs_metadata, 
                   scale=1.0, 
                   instance_mode=ColorMode.IMAGE   
    )

    v = v.draw_instance_predictions(final_output["instances"].to("cpu"))
    im_post_detection = v.get_image()[:, :, ::-1]
    
    
    boxes_out = coco_prop_rpn_processed[0]['proposals'].get('proposal_boxes').tensor.clone()
    boxes_scores = torch.sigmoid(coco_prop_rpn_processed[0]['proposals'].get('objectness_logits').clone()) 
    boxes_out = boxes_out[ood_indices, :]  
    boxes_scores = boxes_scores[ood_indices]
    boxes_after_nms_inds = torchvision.ops.nms(boxes_out, boxes_scores, 0.5)
    ood_coords = boxes_out[boxes_after_nms_inds].cpu().numpy()
    im_new = np.ones((im.shape[0]*2, im.shape[1], im.shape[2]), dtype='uint8')
    im_new[0:im.shape[0], :,:] = im
    
    im = im_post_detection

    ### filtering OODs
    detected_objects = final_output['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy()

    ood_coords = filter_wrong_oods(detected_objects, ood_coords)


    for box_ind in range(ood_coords.shape[0]):
        start_point = (ood_coords[box_ind][0], ood_coords[box_ind][1])
        end_point = (ood_coords[box_ind][2], ood_coords[box_ind][3])       
        # Blue color in BGR 
        color = (255, 0, 0) 
        # Line thickness of 2 px 
        thickness = 4
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        im = cv2.rectangle(im, start_point, end_point, color, thickness) 

    # import ipdb; ipdb.set_trace()
    if isinstance(im, cv2.UMat):
        im = im.get()

    im_new[im.shape[0]:2*im.shape[0], :,:] = im        

    cv2.imwrite('/network/tmp1/bhattdha/detectron2_cityscapes/resnet-101_cityscapes/ood_outputs_original_RPN/' + str(ind).zfill(6) + '.png', im_new) 
    # import ipdb; ipdb.set_trace()
import ipdb; ipdb.set_trace()