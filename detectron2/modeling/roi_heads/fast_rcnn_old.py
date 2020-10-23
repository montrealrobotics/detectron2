# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from numpy.random import default_rng
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from copy import deepcopy
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

logger = logging.getLogger(__name__)

curr_iteration = 0

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(boxes, scores, sigma, image_shapes, score_thresh, nms_thresh, topk_per_image):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """

    
    ## if we have deterministic object detector, sigma would be none. In that case, to let the computation
    ## happen without any issue, we just create copy of boxes. 
    # import pdb; pdb.set_trace()
    if sigma == None:  ## None when deterministic object detection, so let's make sigma 0
        sigma = tuple([v.detach().clone() for v in boxes])
        for i in range(len(sigma)):
            sigma[i][:,:] = 0.0

    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, sigma_per_image,image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, sigma_per_image,image_shape in zip(scores, boxes, sigma, image_shapes)
    ]
    return tuple(list(x) for x in zip(*result_per_image))


def fast_rcnn_inference_single_image(
    boxes, scores, sigma, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    scores = scores[:, :-1]
    new_scores = scores 
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    sigma = sigma.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
        sigma = sigma[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
        sigma = sigma[filter_mask]
        true_indices = torch.where(filter_mask)
        ## send all scores so it can be used to compute PDQ
        all_scores = new_scores[true_indices[0]]
    scores = scores[filter_mask]

    ## normalize for PDQ
    all_scores = all_scores / torch.sum(all_scores, dim = 1).view(len(all_scores), 1)

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, sigma, filter_inds = boxes[keep], scores[keep], sigma[keep],filter_inds[keep]

    ## select here also
    all_scores = all_scores[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_sigma = sigma
    result.pred_classes = filter_inds[:, 1]
    result.all_scores = all_scores
    return result, filter_inds[:, 0]

def getMahaThreshold(prob_val):
    """
        Here, we get probability thresholds,
        let's find corresponding mahalanobis distance
    """
    from scipy.stats import chi2

    ## degrees of freedom, we have bounding box with 4 coordinates, hence 4
    df = 4 

    ## the thresholds of mahalanobis distance will be stored here!
    mahadist_thresh = chi2.ppf(prob_val, df)

    return mahadist_thresh

# def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image):
#     """
#     Call `fast_rcnn_inference_single_image` for all images.

#     Args:
#         boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
#             boxes for each image. Element i has shape (Ri, K * 4) if doing
#             class-specific regression, or (Ri, 4) if doing class-agnostic
#             regression, where Ri is the number of predicted objects for image i.
#             This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
#         scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
#             Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
#             for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
#         image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
#         score_thresh (float): Only return detections with a confidence score exceeding this
#             threshold.
#         nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
#         topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
#             all detections.

#     Returns:
#         instances: (list[Instances]): A list of N instances, one for each image in the batch,
#             that stores the topk most confidence detections.
#         kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
#             the corresponding boxes/scores index in [0, Ri) from the input, for image i.
#     """
#     result_per_image = [
#         fast_rcnn_inference_single_image(
#             boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
#         )
#         for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
#     ]
#     return tuple(list(x) for x in zip(*result_per_image))


# def fast_rcnn_inference_single_image(
#     boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
# ):
#     """
#     Single-image inference. Return bounding-box detection results by thresholding
#     on scores and applying non-maximum suppression (NMS).

#     Args:
#         Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
#         per image.

#     Returns:
#         Same as `fast_rcnn_inference`, but for only one image.
#     """
#     scores = scores[:, :-1]
#     num_bbox_reg_classes = boxes.shape[1] // 4
#     # Convert to Boxes to use the `clip` function ...
#     boxes = Boxes(boxes.reshape(-1, 4))
#     boxes.clip(image_shape)
#     boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

#     # Filter results based on detection scores
#     filter_mask = scores > score_thresh  # R x K
#     # R' x 2. First column contains indices of the R predictions;
#     # Second column contains indices of classes.
#     filter_inds = filter_mask.nonzero()
#     if num_bbox_reg_classes == 1:
#         boxes = boxes[filter_inds[:, 0], 0]
#     else:
#         boxes = boxes[filter_mask]
#     scores = scores[filter_mask]

#     # Apply per-class NMS
#     keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
#     if topk_per_image >= 0:
#         keep = keep[:topk_per_image]
#     boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

#     result = Instances(image_shape)
#     result.pred_boxes = Boxes(boxes)
#     result.scores = scores
#     result.pred_classes = filter_inds[:, 1]
#     return result, filter_inds[:, 0]

curr_iteration = 0
curr_weight_index = 0
dist_save = 0

class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """


    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, pred_proposal_uncertain, proposals, smooth_l1_beta, loss_type, total_iterations
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            loss_type (string): determines what type of loss to apply
                could be ['smooth_l1', 'loss_att', 'cal_loss', 'mahalanobis_attenuation']
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.pred_proposal_uncertain = pred_proposal_uncertain
        self.loss_type = loss_type
        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]
        self.total_iterations = total_iterations
        self.annealing_weights = np.arange(0.01, 0.21, 0.01)

        global curr_iteration
        curr_iteration = curr_iteration + 1 
        self.curr_iteration = curr_iteration
        self.curr_weight_index = 0
        self.increment_val = int(self.total_iterations / (len(list(self.annealing_weights))))
        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
            storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def loss_calibration(self):
        """
        Loss calibration implementation

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        ### 

        ## Computing the loss attenuation
        error_loss = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols] - (self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).sum()
        loss_cal_final = (((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2/(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]) + torch.log(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols])).sum() + error_loss)/self.gt_classes.numel() 

        return loss_cal_final

    def mahalanobis_loss_attenuation(self):
        """
        Loss attenuation with Mahalanobis bound for calibreated uncertainty estiamtion

        Returns:
            scalar Tensor
        """

        ## weight annealing scheme
        curr_weight_index = np.min([int(self.curr_iteration / self.increment_val), len(self.annealing_weights) - 1])
        annealing_weight = self.annealing_weights[curr_weight_index]
        if annealing_weight > 1.0:
            annealing_weight = 1.0
        print("Annealing weight is: ", annealing_weight)
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)



        ### 
        mahathresh = getMahaThreshold(0.7)

        ## Computing the loss attenuation
        maha_dists = (self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2/(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols])  
        loss_attenuation_final = (maha_dists / 2.0 + torch.log(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols])).sum()

        ## Computing mahalanobis penalty   
        maha_dists = maha_dists.sum(dim = 1)
        ## if mahadistance is already less than the threshold, we don't need to worry, so we clamp it to 0. As it's already in our desired range. SO no penalty
        ## hence F.relu().
        mahalanobis_penalty = ((F.relu(maha_dists - mahathresh))).sum()   
        # if mahalanobis_penalty > 100:
        #     import ipdb; ipdb.set_trace()
        print("Mahalanobis penalty is: ", mahalanobis_penalty / self.gt_classes.numel())
        mahalanobis_attenuation_loss = (loss_attenuation_final + annealing_weight * mahalanobis_penalty) / self.gt_classes.numel()
        return mahalanobis_attenuation_loss

    def loss_attenuation(self):
        """
        Loss attenuation implementation

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        ### 

        ##############################################################################################################################################################################

        ## very dangerous piece of code, do not uncomment it without expert supervision

        # our_imp_stuff = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2/(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]))
        # our_stuff = our_imp_stuff.detach().clone().cpu().numpy()
        # global dist_save
        # if dist_save is 0:
        #     dist_save = our_stuff
        # else: 
        #     dist_save = np.concatenate((dist_save, our_stuff),axis=0)

        # if self.curr_iteration == 1000:
        #     print("The shape of dist_save is: {}".format(dist_save.shape))
        #     np.save(f'/home/mila/b/bhattdha/detectron2/unigaussians.npy', np.array(dist_save))
        #     import sys; sys.exit(0)


        ##############################################################################################################################################################################

        ## Computing the loss attenuation
        loss_attenuation_final = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2/(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]) + torch.log(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols])).sum()/self.gt_classes.numel()

        return loss_attenuation_final


    def kl_divergence_batch_loss(self):
        """
        Apply KL divergence loss over the batch

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        preds = self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols].flatten()
        gts = gt_proposal_deltas[fg_inds].flatten()
        variance = self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols].flatten()

        rng = default_rng()
        dof = 100
        no_samples = 1000

        # assert len(preds) > dof*no_samples, 'DoF*no_samples and len(preds) are {} {}'.format(dof*no_samples, len(preds))

        chi_sq_samples = []

        for i in range(no_samples):            
            indices = rng.choice(len(preds), size = dof, replace = False)
            chi_sq_variable = (preds[indices] - gts[indices])**2 / variance[indices]
            chi_sq_samples.append(chi_sq_variable.sum())

        chi_sq_samples = torch.stack(chi_sq_samples)

        emp_mean = chi_sq_samples.mean()
        emp_var = chi_sq_samples.var()
        gt_mean = dof
        gt_variance = 2*dof

        mu1 = gt_mean
        mu2 = emp_mean
        var1 = gt_variance
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        # kldivergence = (torch.log(var2 / var1) + ( var1 + (mu1 - mu2)**2 ) / var2)/2.0
        # kldivergence = (torch.log(var1 / var2) + ( var2 + (mu2 - mu1)**2 ) / var1)/2.0 + (torch.log(var2 / var1) + ( var1 + (mu1 - mu2)**2 ) / var2)/2.0
        # bhattcharya_distance = 0.25 * torch.log(0.25 * (var1/var2 + var2/var1 + 2)) + 0.25 * ((mu1 - mu2)**2/(var1 + var2))
        kldivergence = torch.distributions.kl.kl_divergence(our_dist, actual_dist) / dof

        # if kldivergence > 50:
        # kldivergence = kldivergence / dof ## just normalizing

        # if kldivergence < 0.005:
        #     kldivergence = kldivergence * len(preds) ## just normalizing

        return kldivergence

    def kl_batch_plus_loss_att(self):
        """
        Apply KL divergence + loss attenuation over the batch

        Returns:
            scalar Tensor
        """


        gt_proposal_deltas = self.box2box_transform.get_deltas(self.proposals.tensor, self.gt_boxes.tensor)
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)


        ## loss attenuation 
        loss_attenuation_final = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2/(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]) + torch.log(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols])).sum()/self.gt_classes.numel()

        ## Computing KL-divergence
        preds = self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols].flatten()
        gts = gt_proposal_deltas[fg_inds].flatten()
        variance = self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols].flatten()

        rng = default_rng()
        dof = 100
        no_samples = 1000

        # assert len(preds) > dof*no_samples, 'DoF*no_samples and len(preds) are {} {}'.format(dof*no_samples, len(preds))

        chi_sq_samples = []

        for i in range(no_samples):            
            indices = rng.choice(len(preds), size = dof, replace = False)
            chi_sq_variable = (preds[indices] - gts[indices])**2 / variance[indices]
            chi_sq_samples.append(chi_sq_variable.sum())

        chi_sq_samples = torch.stack(chi_sq_samples)

        emp_mean = chi_sq_samples.mean()
        emp_var = chi_sq_samples.var()
        gt_mean = dof
        gt_variance = 2*dof

        mu1 = gt_mean
        mu2 = emp_mean
        var1 = gt_variance
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        # kldivergence = (torch.log(var2 / var1) + ( var1 + (mu1 - mu2)**2 ) / var2)/2.0
        # kldivergence = (torch.log(var1 / var2) + ( var2 + (mu2 - mu1)**2 ) / var1)/2.0 + (torch.log(var2 / var1) + ( var1 + (mu1 - mu2)**2 ) / var2)/2.0
        # bhattcharya_distance = 0.25 * torch.log(0.25 * (var1/var2 + var2/var1 + 2)) + 0.25 * ((mu1 - mu2)**2/(var1 + var2))
        kldivergence = torch.distributions.kl.kl_divergence(our_dist, actual_dist) / dof

        # if kldivergence > 50:
        # kldivergence = kldivergence / dof ## just normalizing

        # if kldivergence < 0.005:
        #     kldivergence = kldivergence * len(preds) ## just normalizing

        return kldivergence + loss_attenuation_final

    def kl_batch_plus_smoothl1(self):
        """
        Apply KL divergence + smoothl1 over the batch

        Returns:
            scalar Tensor
        """

        curr_weight_index = np.min([int(self.curr_iteration / self.increment_val), len(self.annealing_weights) - 1])
        annealing_weight =  0.01
        print("Annealing weight is: {}".format(annealing_weight))
        # if annealing_weight > 1.0:
        #     annealing_weight = 1.0

        gt_proposal_deltas = self.box2box_transform.get_deltas(self.proposals.tensor, self.gt_boxes.tensor)
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)


        ##############################################################################################################################################################################

        # very dangerous piece of code, do not uncomment it without expert supervision

        # our_imp_stuff = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2/(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]))
        # our_stuff = our_imp_stuff.detach().clone().cpu().numpy()
        # global dist_save
        # if dist_save is 0:
        #     dist_save = our_stuff
        # else: 
        #     dist_save = np.concatenate((dist_save, our_stuff),axis=0)

        # if self.curr_iteration == 500:
        #     print("The shape of dist_save is: {}".format(dist_save.shape))
        #     np.save(f'/network/tmp1/bhattdha/detectron2_cityscapes/kl_plus_loss_att_frozen/unigaussians_kl_plus_loss_att_frozen.npy', np.array(dist_save))
        #     import sys; sys.exit(0)

        ##############################################################################################################################################################################

        # smooth l1
        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        loss_box_reg = loss_box_reg / self.gt_classes.numel()


        ## Computing KL-divergence
        preds = self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols].flatten()
        gts = gt_proposal_deltas[fg_inds].flatten()
        variance = self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols].flatten()

        rng = default_rng()
        dof = 100
        no_samples = 1000

        # assert len(preds) > dof*no_samples, 'DoF*no_samples and len(preds) are {} {}'.format(dof*no_samples, len(preds))

        chi_sq_samples = []

        for i in range(no_samples):            
            indices = rng.choice(len(preds), size = dof, replace = False)
            chi_sq_variable = (preds[indices] - gts[indices])**2 / variance[indices]
            chi_sq_samples.append(chi_sq_variable.sum())

        chi_sq_samples = torch.stack(chi_sq_samples)

        emp_mean = chi_sq_samples.mean()
        emp_var = chi_sq_samples.var()
        gt_mean = dof
        gt_variance = 2*dof

        mu1 = gt_mean
        mu2 = emp_mean
        var1 = gt_variance
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        # kldivergence = (torch.log(var2 / var1) + ( var1 + (mu1 - mu2)**2 ) / var2)/2.0
        # kldivergence = (torch.log(var1 / var2) + ( var2 + (mu2 - mu1)**2 ) / var1)/2.0 + (torch.log(var2 / var1) + ( var1 + (mu1 - mu2)**2 ) / var2)/2.0
        # bhattcharya_distance = 0.25 * torch.log(0.25 * (var1/var2 + var2/var1 + 2)) + 0.25 * ((mu1 - mu2)**2/(var1 + var2))
        


        kldivergence = torch.distributions.kl.kl_divergence(our_dist, actual_dist) * annealing_weight
        # if kldivergence > 50:
        # kldivergence = kldivergence / dof ## just normalizing

        # if kldivergence < 0.005:
        #     kldivergence = kldivergence * len(preds) ## just normalizing

        print("KL divergence, smooth_l1 losses and current itrations are {}, {} and {}".format(kldivergence, loss_box_reg, self.curr_iteration))

        return kldivergence + loss_box_reg
        
    def wasserstein_batch_plus_smoothl1(self):
        """
        Apply KL divergence + loss attenuation over the batch

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(self.proposals.tensor, self.gt_boxes.tensor)
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)


        ##############################################################################################################################################################################

        ## very dangerous piece of code, do not uncomment it without expert supervision

        # our_imp_stuff = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2/(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]))
        # our_stuff = our_imp_stuff.detach().clone().cpu().numpy()
        # global dist_save
        # if dist_save is 0:
        #     dist_save = our_stuff
        # else: 
        #     dist_save = np.concatenate((dist_save, our_stuff),axis=0)

        # if self.curr_iteration == 500:
        #     print("The shape of dist_save is: {}".format(dist_save.shape))
        #     np.save(f'/home/mila/b/bhattdha/detectron2/unigaussians_loss_att_model_cityscapes.npy', np.array(dist_save))
        #     import sys; sys.exit(0)

        ##############################################################################################################################################################################

        # ## smooth l1
        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        loss_box_reg = loss_box_reg / self.gt_classes.numel()


        ## Computing KL-divergence
        preds = self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols].flatten()
        gts = gt_proposal_deltas[fg_inds].flatten()
        variance = self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols].flatten()

        rng = default_rng()
        dof = 100
        no_samples = 1000

        # assert len(preds) > dof*no_samples, 'DoF*no_samples and len(preds) are {} {}'.format(dof*no_samples, len(preds))

        chi_sq_samples = []

        for i in range(no_samples):            
            indices = rng.choice(len(preds), size = dof, replace = False)
            chi_sq_variable = (preds[indices] - gts[indices])**2 / variance[indices]
            chi_sq_samples.append(chi_sq_variable.sum())

        chi_sq_samples = torch.stack(chi_sq_samples)

        emp_mean = chi_sq_samples.mean()
        emp_var = chi_sq_samples.var()
        gt_mean = dof
        gt_variance = 2*dof

        mu1 = gt_mean
        mu2 = emp_mean
        var1 = gt_variance
        var2 = emp_var

        ## wasserstein distance between two Gaussians
        wasserstein_distance = ((mu1 - mu2)**2 + var1 + var2 - 2*((var1*var2).sqrt())) / dof**2

        print("Wasserstein distance, smooth_l1 losses and current itrations are {}, {} and {}".format(wasserstein_distance, loss_box_reg, self.curr_iteration))

        return wasserstein_distance + loss_box_reg


    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        # import pdb; pdb.set_trace()
        return loss_box_reg


    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and regression loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_attenuation_final".
        """

        if self.loss_type == 'smooth_l1':
            loss_name = 'smooth_l1_loss'
            loss_reg = self.smooth_l1_loss()
        elif self.loss_type == 'loss_att':
            loss_name = 'loss_attenuation'
            loss_reg = self.loss_attenuation()
        elif self.loss_type == 'loss_cal':
            loss_name = 'loss_calibration'
            loss_reg = self.loss_calibration()
        elif self.loss_type == 'mahalanobis_attenuation':
            loss_name = 'mahalanobis_loss_attenuation'
            loss_reg = self.mahalanobis_loss_attenuation()
        elif self.loss_type == 'kl_divergence_batch_loss':
            loss_name = 'kl_divergence_batch_loss'
            loss_reg = self.kl_divergence_batch_loss()
        elif self.loss_type == 'kl_batch_plus_loss_att':
            loss_name = 'kl_batch_plus_loss_att'
            loss_reg = self.kl_batch_plus_loss_att()
        elif self.loss_type == 'kl_batch_plus_smoothl1':
            loss_name = 'kl_batch_plus_smoothl1'
            loss_reg = self.kl_batch_plus_smoothl1()
        elif self.loss_type == 'wasserstein_batch_plus_smoothl1':
            loss_name = 'wasserstein_batch_plus_smoothl1'
            loss_reg = self.wasserstein_batch_plus_smoothl1()

            # loss_reg = self.loss_attenuation()

        return {
            loss_name: loss_reg,
            "loss_cls": self.softmax_cross_entropy_loss()
        }

        # return {
        #     "loss_cls": self.softmax_cross_entropy_loss(),
        #     "loss_box_reg": self.smooth_l1_loss(),
        # }

    def predict_variance(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_uncertain.shape[1] // B
        boxes_var = self.box2box_transform.apply_deltas_variance(
            self.pred_proposal_uncertain.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes_var.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        if self.loss_type != 'smooth_l1':
            sigma = self.predict_variance()
        else: 
            sigma = None
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return fast_rcnn_inference(
            boxes, scores, sigma, image_shapes, score_thresh, nms_thresh, topk_per_image
        )


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4, cfg=None):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
            cfg: config file for type of detector(deterministic or probabilitic)
        """
        super(FastRCNNOutputLayers, self).__init__()
        self.cfg = cfg
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)

        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)
        if cfg is not None: 
            if cfg.CUSTOM_OPTIONS.DETECTOR_TYPE == 'probabilistic': 
                #### Adding uncertainty prediction(not the best way but a good start) ####
                self.bbox_uncertainty_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
                nn.init.normal_(self.bbox_uncertainty_pred.weight, std=0.001)
                nn.init.constant_(self.bbox_uncertainty_pred.bias, 0)


    def RichardCurve(self, x, low=0, high=1, sharp=0.5):
        r"""Applies the generalized logistic function (aka Richard's curve)
        to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor over which the generalized logistic
                function is to be applied (independently over each element)
            low (float): Lower asymptote of the Richard's curve
            high (float): Upper asymptote of the Richard's curve
            sharp (float): Controls the 'sharpness' of the slope for the linear
                region of Richard's curve

        """
        return low + ((high - low) / (1 + torch.exp(-sharp * x)))

    def forward(self, x):
        
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)

        ## Because uncertainty is always +ve
        # proposal_delta_uncertainty = self.RichardCurve(self.bbox_uncertainty_pred(x), low=0, high=10)

        if self.cfg is not None:
            if self.cfg.CUSTOM_OPTIONS.DETECTOR_TYPE == 'probabilistic': 
                # proposal_delta_uncertainty = self.RichardCurve(self.bbox_uncertainty_pred(x), low=1e-3, high=10, sharp=0.15) ## This was used for the model that works
                proposal_delta_uncertainty = self.RichardCurve(self.bbox_uncertainty_pred(x), low=self.cfg.CUSTOM_OPTIONS.RICHARD_CURVE_LOW, high=self.cfg.CUSTOM_OPTIONS.RICHARD_CURVE_HIGH, sharp=self.cfg.CUSTOM_OPTIONS.RICHARD_CURVE_SHARP) ## This was used for the model that works
                
                return scores, proposal_deltas, proposal_delta_uncertainty
            else:
                return scores, proposal_deltas, None
        # import pdb; pdb.set_trace()
        return scores, proposal_deltas, None
        