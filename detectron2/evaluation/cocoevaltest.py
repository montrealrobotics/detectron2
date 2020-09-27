from cocoLRP import COCOevalLRP
from pycocotools.coco import COCO
import json

gt_file = '/network/tmp1/bhattdha/coco/annotations/instances_val2017.json'
cocoGt=COCO(gt_file)

res_file = '/home/mila/b/bhattdha/detectron2/coco_all_classes_loss_attenuation_model_0051999.pth/coco_instances_results.json'
cocoDt=cocoGt.loadRes(res_file)

tau=0.5
DetailedLRPResultNeeded=0
cocoEvalLRP = COCOevalLRP(cocoGt,cocoDt,tau)
cocoEvalLRP.evaluate()
cocoEvalLRP.accumulate()
cocoEvalLRP.summarize(DetailedLRPResultNeeded)
