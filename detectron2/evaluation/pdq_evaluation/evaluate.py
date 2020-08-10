from .read_files import *
import argparse
import os
import sys
from .pdq import PDQ
import json
import numpy as np


class ParamSequenceHolder:
    def __init__(self, gt_instances_lists, det_instances_lists):
        """
        Class for holding parameters (GroundTruthInstances etc.) for multiple sequences.
        Based upon match_sequences function from codalab challenge but with fewer checks.
        Link to codalab challenge version: https://github.com/jskinn/rvchallenge-evaluation/blob/master/gt_loader.py
        :param gt_instances_lists: list of gt_instance_lists (one gt_instance_list per sequence)
        :param det_instances_lists: list of det_instance_lists (one det_instance_list per sequence)
        Note, order of gt_instances_list and det_instances_list must be the same (corresponding sequences)

        """
        self._gt_instances_lists = gt_instances_lists
        self._det_instances_lists = det_instances_lists

    def __len__(self):
        length = np.sum([len(gt_list) for gt_list in self._gt_instances_lists])
        return length

    def __iter__(self):

        for idx in range(len(self._gt_instances_lists)):
            gt_list = self._gt_instances_lists[idx]
            det_list = self._det_instances_lists[idx]

            # Check the lists are the same length
            if len(gt_list) != len(det_list):
                raise ValueError('gt_list and det_list for sequence {0} not the same length\n'
                                 'length GT: {1}\n'
                                 'length Det {2}'.format(idx, len(gt_list), len(det_list)))

            for frame_gt, frame_detections in zip(gt_list, det_list):
                ground_truth = list(frame_gt)
                detections = list(frame_detections)
                yield ground_truth, detections


def gen_param_sequence(gt_coco_json_path, det_json_path):
    """
    Function for generating the parameter sequence to be used in evaluation procedure.
    Parameter sequence holds all GroundTruthInstances, DetectionInstances, and ground-truth filter flags
    across all sequences.
    :return: param_sequences: ParamSequenceHolder containing all GroundTruthInstances, DetectionInstances,
    and ground-truth filter flags across all sequences being evaluated.
    len_sequences: list of sequence lengths for all sequences being evaluated.
    """

    # Load GTs and Detections as appropriate for different data sets (multiple sequences or one folder)

    # output is a generator of lists of GTInstance objects and a map of gt_class_ids
    gt_instances, gt_class_ids = read_COCO_gt(gt_coco_json_path, ret_classes=True, bbox_gt=False)

    # output is a generator of lists of DetectionInstance objects (BBox or PBox depending on detection)
    det_instances = read_pbox_json(det_json_path, gt_class_ids, override_cov=None,
                                              prob_seg=False)
    all_gt_instances = [gt_instances]
    all_det_instances = [det_instances]


    param_sequence = ParamSequenceHolder(all_gt_instances, all_det_instances)
    len_sequences = [len(all_gt_instances[idx]) for idx in range(len(all_gt_instances))]

    return param_sequence, len_sequences


def evaluation_PDQ(gt_coco_json_path, det_json_path):
    
    print("Extracting GT and Detections")
    param_sequence, len_sequences = gen_param_sequence(gt_coco_json_path, det_json_path)

    print("Calculating PDQ")

    # Get summary statistics (PDQ, avg_qualities)
    evaluator = PDQ(filter_gts=False, segment_mode=False, greedy_mode=False)
    pdq = evaluator.score(param_sequence)
    TP, FP, FN = evaluator.get_assignment_counts()
    avg_spatial_quality = evaluator.get_avg_spatial_score()
    avg_label_quality = evaluator.get_avg_label_score()
    avg_overall_quality = evaluator.get_avg_overall_quality_score()
    
    # Get the detection-wise and ground-truth-wise qualities and matches for PDQ and save them to file
    all_gt_eval_dicts = evaluator._gt_evals
    all_det_eval_dicts = evaluator._det_evals

    result = {"PDQ": pdq, "avg_pPDQ": avg_overall_quality, "avg_spatial": avg_spatial_quality,
              "avg_label": avg_label_quality, "TP": TP, "FP": FP, "FN": FN}

    print("PDQ: {0:4f}\n"
          "avg_pPDQ:{1:4f}\n"
          "avg_spatial:{2:4f}\n"
          "avg_label:{3:4f}\n"
          "TP:{4}\nFP:{5}\nFN:{6}\n".format(pdq, avg_overall_quality, avg_spatial_quality, avg_label_quality, TP, FP, FN))

    return result