# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from numpy.random import default_rng
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F
import os
from copy import deepcopy
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

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

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)


## calculate expected calibration error
# def ECE(x, mu, std_dev):
#     emp_ps = []
#     ece = []
#     N = len(x)
#     probs = np.arange(0.0 , 1.005, 0.005)
#     for p in probs:
#         N_p = ((x-mu)/std_dev < norm.ppf(p)).sum()
#         emp_p = N_p/len(x)
#         emp_ps.append(emp_p)
#         ece.append(N_p/N*abs(emp_p-p))  
#     ece = np.sum(ece)  
#     return ece, emp_ps, probs

## calculate expected calibration error
def ECE(x, mu, std_dev):
    eps = 0.005
    emp_ac_ps = []
    ECE = []
    cal_curve_stats = []
    N = len(x)
    k = 0
    list_p = np.arange(0,1.0+eps, eps)
    for p in list_p:
        N_p = ((x-mu)/std_dev < norm.ppf(p)).sum()
        N_b = np.logical_and((x-mu)/std_dev < norm.ppf(p), (x-mu)/std_dev > norm.ppf(k)).sum()
        emp_p = N_b / N
        emp_ac_ps.append(N_p / N)
        ECE.append((N_b / N) * abs(emp_p - eps))
        k = p
    ece = np.sum(ECE) 
    return ece, emp_ac_ps, list_p

def store_and_plot_residuals(residual_variable = None, cfg = None, data_dict = None):


    #######################################
    ## This is a helper function to store and plot
    ## chi-sqaured corresponding to the residuals collected!
    #######################################

    assert residual_variable is not None
    assert cfg is not None
    assert data_dict is not None    

    dists = residual_variable.flatten()
    mean_dist = dists.mean()
    variance_dist = dists.var()
    
    np.random.seed(42)
    ece, emp_ps, probs = ECE( data_dict['gt'], data_dict['mu'], data_dict['std_dev'])
    ece_sn_dict = {'ece_sn':ece, 'emp_ps_sn':emp_ps, 'probs':probs}
    plt.plot(probs, emp_ps, 'k', linewidth=2)
    title = "Fit results: ECE = %.2f" % (ece)
    plt.title(title)
    plt.savefig(os.path.join(cfg.CUSTOM_OPTIONS.RESIDUAL_DIR_NAME, cfg.CUSTOM_OPTIONS.MODEL_NAME + '_sn_reliability_diagram.png')) 
    plt.clf()

    mu, std = norm.fit(dists)
    print("Mean and variance are: ", mu, std**2)

    plt.hist(dists, bins=100, density=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  variance = %.2f" % (mu, std**2)
    plt.title(title)
    plt.savefig(os.path.join(cfg.CUSTOM_OPTIONS.RESIDUAL_DIR_NAME, cfg.CUSTOM_OPTIONS.MODEL_NAME + '_standard_normal.png')) 
    plt.clf()

    df = 100
    dists = dists**2
    dists = np.random.permutation(dists)

    samples = []

    for i in range(1):
        dists = np.random.permutation(dists)
        for j in np.arange(0, len(dists) - df, df):
            # print('i is: ', i)
            dist_vals = dists[j:(j+df)]
            samples.append(dist_vals.sum())

    samples = np.array(samples)
    data = samples

    ece_cs, emp_ps_cs, probs_cs = ECE(samples, df, (2*df)**0.5)
    ece_cs_dict = {'ece_cs': ece_cs, 'emp_ps_cs': emp_ps_cs, 'probs_cs':probs_cs}
    plt.plot(probs_cs, emp_ps_cs, 'k', linewidth=2)
    title = "Fit results: ECE = %.2f" % (ece_cs)
    plt.title(title)
    plt.savefig(os.path.join(cfg.CUSTOM_OPTIONS.RESIDUAL_DIR_NAME, cfg.CUSTOM_OPTIONS.MODEL_NAME + '_cs_reliability_diagram.png')) 
    plt.clf()

    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    print("Mean and variance are: ", mu, std**2)
    # Plot the histogram.
    plt.hist(data, bins=100, density=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.savefig(os.path.join(cfg.CUSTOM_OPTIONS.RESIDUAL_DIR_NAME, cfg.CUSTOM_OPTIONS.MODEL_NAME + '_chi_squared.png')) 
    mu1 = df
    mu2 = mu
    var1 = 2*df
    var2 = std**2        

    print("Emp mean and emp variance are {} {}".format(mu2, var2))
    wasserstein_distance = ((mu1 - mu2)**2 + var1 + var2 - 2*(var1*var2)**0.5) 

    our_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
    actual_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))
    kldivergence = torch.distributions.kl.kl_divergence(actual_dist, our_dist) 
    dists = {'kld': kldivergence, 'wdist': wasserstein_distance, 'emp_mean': mu, 'emp_var': std**2, 'ECE_sn': ece_sn_dict, 'ECE_cs': ece_cs_dict}
    print(dists)
    data_dict['dists'] = dists
    np.save(os.path.join(cfg.CUSTOM_OPTIONS.RESIDUAL_DIR_NAME, cfg.CUSTOM_OPTIONS.MODEL_NAME + '.npy'), np.array(dist_save))
    np.save(os.path.join(cfg.CUSTOM_OPTIONS.RESIDUAL_DIR_NAME, cfg.CUSTOM_OPTIONS.MODEL_NAME + '_' + cfg.DATASETS.TRAIN[0] + '_all_data' + '.npy'), data_dict) 
    return 0


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
mu = 0
sigma = 0
gt = 0

class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """


    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, pred_proposal_uncertain, proposals, smooth_l1_beta, loss_type, total_iterations, cfg = None
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
        self.cfg = cfg
        assert self.cfg is not None
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
        error_loss = smooth_l1_loss(
            self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols],
            (self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2,
            1,
            reduction="sum",
        )
        loss_cal_final = (((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2/(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]) + torch.log(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols])).sum() + error_loss)/self.gt_classes.numel() 

        
        ########################################################################################################################################################################################################################    
        ###### In loss attenuation, we do keep track of mean and variance of chi-squared

        preds = self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols].flatten()
        gts = gt_proposal_deltas[fg_inds].flatten()
        variance = self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols].flatten()

        rng = default_rng()
        dof = 75
        no_samples = 100

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

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        kldivergence = torch.distributions.kl.kl_divergence(actual_dist, our_dist) * 0.2
        print("kl_div_chi_sq_closed_form is: {}".format(kldivergence))

        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term, kl_divergence = kldivergence)
        ########################################################################################################################################################################################################################

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


        ########################################################################################################################################################################################################################    
        ###### In loss attenuation, we do keep track of mean and variance of chi-squared

        preds = self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols].flatten()
        gts = gt_proposal_deltas[fg_inds].flatten()
        variance = self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols].flatten()

        rng = default_rng()
        dof = 75
        no_samples = 100

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

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        kldivergence = torch.distributions.kl.kl_divergence(actual_dist, our_dist) * 0.2
        print("kl_div_chi_sq_closed_form is: {}".format(kldivergence))

        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term, kl_divergence = kldivergence)
        ########################################################################################################################################################################################################################

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

    def variance_loss(self):
        """
        variance loss

        try to match sigma^2 with (x - mu)^2

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
        
        variance_loss = smooth_l1_loss(variance, (gts - preds)**2, self.smooth_l1_beta, reduction="sum")

        variance_loss = variance_loss / self.gt_classes.numel()
        print("Variance loss is : {}".format(variance_loss))
        return variance_loss

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


        ## Computing the loss attenuation
        loss_attenuation_final = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2/(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]) + torch.log(self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols])).sum()/self.gt_classes.numel()


        ######################################################################################################################################################################################################################## 

        ###### In loss attenuation, we do keep track of mean and variance of chi-squared

        preds = self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols].flatten()
        gts = gt_proposal_deltas[fg_inds].flatten()
        variance = self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols].flatten()

        rng = default_rng()
        dof = 75
        no_samples = 100

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

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        kldivergence = torch.distributions.kl.kl_divergence(actual_dist, our_dist) * 0.2
        print("kl_div_chi_sq_closed_form is: {}".format(kldivergence))

        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term, kl_divergence = kldivergence)
        ########################################################################################################################################################################################################################

        return loss_attenuation_final

    def collect_training_stats(self):
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

        preds_numpy = preds.detach().clone().cpu().numpy()
        gts_numpy = gts.cpu().numpy()
        std_dev_numpy = (variance**0.5).detach().clone().cpu().numpy()

        global mu
        global sigma
        global gt

        if mu is 0:
            mu = preds_numpy
            sigma = std_dev_numpy
            gt = gts_numpy
        else:
            mu = np.concatenate((mu, preds_numpy),axis=0)
            sigma = np.concatenate((sigma, std_dev_numpy), axis = 0)
            gt = np.concatenate((gt, gts_numpy), axis = 0)

        std_normal_samples = (gts - preds) / variance.sqrt()
        our_stuff = std_normal_samples.detach().clone().cpu().numpy()
        print("shape of residual vector is {}".format(our_stuff.shape))

        global dist_save
        if dist_save is 0:
            dist_save = our_stuff
        else: 
            dist_save = np.concatenate((dist_save, our_stuff),axis=0)

        if self.curr_iteration == self.cfg.CUSTOM_OPTIONS.RESIDUAL_MAX_ITER:
            ## we have to save the residuals now!
            data_dict = {'mu': mu, 'std_dev': sigma, 'gt': gt}
            gt_dict = {'gt':gt}
            mu_dict = {'mu':mu}
            std_dev_dict = {'std_dev': sigma}
            val = store_and_plot_residuals(residual_variable = dist_save, cfg = self.cfg, data_dict = data_dict)
            import sys; sys.exit(0)
            
        return 0

    def collect_residuals(self):
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

        std_normal_samples = (gts - preds) / variance.sqrt()
        our_stuff = std_normal_samples.detach().clone().cpu().numpy()
        print("shape of residual vector is {}".format(our_stuff.shape))

        global dist_save
        if dist_save is 0:
            dist_save = our_stuff
        else: 
            dist_save = np.concatenate((dist_save, our_stuff),axis=0)

        if self.curr_iteration == self.cfg.CUSTOM_OPTIONS.RESIDUAL_MAX_ITER:
            ## we have to save the residuals now!
            print("The shape of dist_save is: {}".format(dist_save.shape))
            val = store_and_plot_residuals(residual_variable = dist_save, cfg = self.cfg)
            np.save(os.path.join(self.cfg.CUSTOM_OPTIONS.RESIDUAL_DIR_NAME, self.cfg.CUSTOM_OPTIONS.MODEL_NAME + '.npy'), np.array(dist_save))
            import sys; sys.exit(0)

        return 0



    def wasserstein_over_chi_squared(self):
        """
        Apply wasserstein distance over the chi-squared distributions

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

        print("Total number of samples are: {}".format(preds.shape))

        rng = default_rng()
        dof = 75
        no_samples = 100

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

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var        

        print("Emp mean and emp variance are {} {}".format(mu2, var2))
        wasserstein_distance = ((mu1 - mu2)**2 + var1 + var2 - 2*(var1*var2)**0.5) * 1e-3
        print("wasserstein loss is {}".format(wasserstein_distance))

        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term)
        storage.put_scalar("wasserstein_loss/chi-squared", wasserstein_distance)

        return wasserstein_distance         

    def wasserstein_over_standard_normal(self):
        """
        Apply wasserstein distance over the standard normal distribution

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

        std_normal_samples = (gts - preds) / variance.sqrt()

        emp_mean = std_normal_samples.mean()
        emp_var = std_normal_samples.var()
        gt_mean = 0
        gt_variance = 1

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var

        print("Emp mean and emp variance are {} {}".format(mu2, var2))
        wasserstein_distance = ((mu1 - mu2)**2 + var1 + var2 - 2*(var1*var2)**0.5) * 10
        print("wasserstein loss is {}".format(wasserstein_distance))

        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term)

        storage.put_scalar("wasserstein_loss/standard-normal", wasserstein_distance)

        return wasserstein_distance         


    def kl_div_CLT_closed_form_gaussian(self):
        """
        Apply KL divergence over standard-normal distribution
        derived from Central limit theorem

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

        print("Total number of samples are: {}".format(preds.shape))

        rng = default_rng()
        sample_size = 40
        no_samples = 400

        # assert len(preds) > dof*no_samples, 'DoF*no_samples and len(preds) are {} {}'.format(dof*no_samples, len(preds))

        mean_samples_vector = []

        for i in range(no_samples):            
            indices = rng.choice(len(preds), size = sample_size, replace = False)
            sample_mean = torch.mean(0.5 * (1 + torch.erf((gts[indices] - preds[indices]) * torch.sqrt(variance[indices]).reciprocal() / np.sqrt(2))))
            mean_samples_vector.append(sample_mean)

        mean_samples_vector = torch.stack(mean_samples_vector)

        ## CDF of large number of samples
        ## look like a uniform distribution
        low = 0.0
        high = 1.0

        ## mean and variance of standard uniform distribution
        ## this results out of interesting property of CLT
        mu_uni = (low + high) / 2.0
        sigma_uni = np.sqrt((high - low)**2 / 12.0)

        new_sample_means = (np.sqrt(sample_size) * (mean_samples_vector * sample_size - torch.mean(mean_samples_vector * sample_size))) / sigma_uni

        emp_mean = new_sample_means.mean()
        emp_var = new_sample_means.var()
        gt_mean = 0
        gt_variance = 1

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var        

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        kldivergence = torch.distributions.kl.kl_divergence(actual_dist, our_dist) * 0.2
        print("kl_div_chi_sq_closed_form is: {}".format(kldivergence))
        
        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term)
        storage.put_scalar("kldivergence/CLT-gaussian", kldivergence)

        return kldivergence

    def kl_div_chi_sq_closed_form(self):
        """
        Apply KL divergence loss over the chi-squared distribution

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
        dof = 75
        no_samples = 100

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

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        kldivergence = torch.distributions.kl.kl_divergence(actual_dist, our_dist) * 0.2
        print("kl_div_chi_sq_closed_form is: {}".format(kldivergence))

        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term)
        storage.put_scalar("KLD-closed-form/chi-squared", kldivergence)

        return kldivergence

    def kl_div_standard_normal_closed_form(self):
        """
        Apply KL divergence loss over standard normal distribution

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

        std_normal_samples = (gts - preds) / variance.sqrt()

        emp_mean = std_normal_samples.mean()
        emp_var = std_normal_samples.var()
        gt_mean = 0
        gt_variance = 1

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        # https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
        kldivergence = torch.distributions.kl.kl_divergence(actual_dist, our_dist) 
        print("kl_div_standard_normal_closed_form is {}".format(kldivergence))


        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term)
        storage.put_scalar("KLD-closed-form/standard-normal", kldivergence)

        return kldivergence

    def kl_div_chi_sq_empirical(self):
        """
        Apply KL divergence loss over the chi-squared distribution(empirical)

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
        dof = 75
        no_samples = 100

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

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        actual_log_probs = our_dist.log_prob(chi_sq_samples)
        true_log_probs = actual_dist.log_prob(chi_sq_samples)

        ### we have to normalize the distribution since it's made from density of continuous distribution samples
        actual_probs = actual_log_probs.exp()
        actual_normalized_log_probs = torch.log(actual_probs / torch.sum(actual_probs))
        true_probs = true_log_probs.exp()
        true_normalized_log_probs = torch.log(true_probs / torch.sum(true_probs))

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        ## refer this to understand arguments of empirical KL divergence!
        ## https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#KLDivLoss
        kldivergence = F.kl_div(actual_normalized_log_probs, true_normalized_log_probs.exp(), reduction = 'mean')
        print("kl_div_chi_sq_empirical is: {}".format(kldivergence))

        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term)
        storage.put_scalar("KLD-empirical/chi-squared", kldivergence)

        return kldivergence

    def kl_div_standard_normal_empirical(self):
        """
        Apply KL divergence loss over the standard normal distribution(empirical)

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

        
        std_normal_samples = (gts - preds) / variance.sqrt()

        emp_mean = std_normal_samples.mean()
        emp_var = std_normal_samples.var()
        gt_mean = 0
        gt_variance = 1

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        actual_log_probs = our_dist.log_prob(std_normal_samples)
        true_log_probs = actual_dist.log_prob(std_normal_samples)

        actual_probs = actual_log_probs.exp()
        actual_normalized_log_probs = torch.log(actual_probs / torch.sum(actual_probs))
        true_probs = true_log_probs.exp()
        true_normalized_log_probs = torch.log(true_probs / torch.sum(true_probs))

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        kldivergence = F.kl_div(actual_normalized_log_probs, true_normalized_log_probs.exp(), reduction = 'sum')
        print("kl_div_standard_normal_empirical is: {}".format(kldivergence))

        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term)        
        storage.put_scalar("KLD-empirical/standard-normal", kldivergence)

        return kldivergence

    def js_div_chi_sq_closed_form(self):
        """
        Apply JS divergence loss over the chi-squared distribution

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
        dof = 75
        no_samples = 100

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

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var

        mu_mix = (mu1 + mu2) / 2.0
        var_mix = (var1 + var2) / 4.0

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5)) ## Q
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5)) ## P
        mix_dist = torch.distributions.normal.Normal(mu_mix, var_mix**(0.5)) ## M


        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        jsdivergence = (torch.distributions.kl.kl_divergence(our_dist, mix_dist) +  torch.distributions.kl.kl_divergence(actual_dist, mix_dist) )/ 200*dof

        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term)
        storage.put_scalar("JSD-closed-form/chi-squared", jsdivergence)

        return jsdivergence

    def js_div_standard_normal_closed_form(self):
        """
        Apply JS divergence loss over standard normal distribution

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

        std_normal_samples = (gts - preds) / variance.sqrt()

        emp_mean = std_normal_samples.mean()
        emp_var = std_normal_samples.var()
        gt_mean = 0
        gt_variance = 1

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var
        mu_mix = (mu1 + mu2) / 2.0
        var_mix = (var1 + var2) / 4.0

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5)) ## Q
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5)) ## P
        mix_dist = torch.distributions.normal.Normal(mu_mix, var_mix**(0.5)) ## M

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        jsdivergence = (torch.distributions.kl.kl_divergence(our_dist, mix_dist) +  torch.distributions.kl.kl_divergence(actual_dist, mix_dist) )/ 2
        print("js_div_standard_normal_closed_form loss is: {}".format(jsdivergence.item()))

        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term)
        storage.put_scalar("JSD-closed-form/standard-normal", jsdivergence)

        return jsdivergence


    def js_div_chi_sq_empirical(self):
        """
        Apply JS divergence loss over the chi-squared distribution(empirical)

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
        dof = 75
        no_samples = 100

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

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        actual_log_probs = our_dist.log_prob(chi_sq_samples)
        true_log_probs = actual_dist.log_prob(chi_sq_samples)
        
        ## getting actual density
        actual_probs = actual_log_probs.exp()
        true_probs = true_log_probs.exp()

        ## normalizing density
        actual_normalized_probs = actual_probs / torch.sum(actual_probs)
        true_normalized_probs = true_probs / torch.sum(true_probs)

        ## getting mixture distribution for chi-squared
        mix_normalized_probs = (actual_normalized_probs + true_normalized_probs) / 2.0

        actual_normalized_log_probs = torch.log(actual_normalized_probs)
        true_normalized_log_probs = torch.log(true_normalized_probs)
        mix_normalized_log_probs = torch.log(mix_normalized_probs)

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        ## refer this to understand arguments of empirical KL divergence!
        ## https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#KLDivLoss
        jsdivergence = (F.kl_div(actual_normalized_log_probs, mix_normalized_log_probs.exp(), reduction = 'sum') +  F.kl_div(actual_normalized_log_probs, mix_normalized_log_probs.exp(), reduction = 'sum')) / 2.0
        print("js_div_chi_sq_empirical is {}".format(jsdivergence.item()))

        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term)
        storage.put_scalar("JSD-empirical/chi-squared", jsdivergence)


        return jsdivergence

    def js_div_standard_normal_empirical(self):
        """
        Apply JS divergence loss over the standard normal distribution(empirical)

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

        
        std_normal_samples = (gts - preds) / variance.sqrt()

        emp_mean = std_normal_samples.mean()
        emp_var = std_normal_samples.var()
        gt_mean = 0
        gt_variance = 1

        if torch.isnan(emp_mean):
            import ipdb; ipdb.set_trace()

        mu1 = gt_mean*torch.ones_like(emp_mean)
        mu2 = emp_mean
        var1 = gt_variance*torch.ones_like(emp_var)
        var2 = emp_var

        actual_dist = torch.distributions.normal.Normal(mu1, var1**(0.5))
        our_dist = torch.distributions.normal.Normal(mu2, var2**(0.5))

        actual_log_probs = our_dist.log_prob(std_normal_samples)
        true_log_probs = actual_dist.log_prob(std_normal_samples)
        
        ## getting actual density
        actual_probs = actual_log_probs.exp()
        true_probs = true_log_probs.exp()

        ## normalizing density
        actual_normalized_probs = actual_probs / torch.sum(actual_probs)
        true_normalized_probs = true_probs / torch.sum(true_probs)

        ## getting mixture distribution for chi-squared
        mix_normalized_probs = (actual_normalized_probs + true_normalized_probs) / 2.0

        actual_normalized_log_probs = torch.log(actual_normalized_probs)
        true_normalized_log_probs = torch.log(true_normalized_probs)
        mix_normalized_log_probs = torch.log(mix_normalized_probs)

        print("Emp mean and emp variance are {} {}".format(mu2, var2))

        ## refer this to understand arguments of empirical KL divergence!
        ## https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#KLDivLoss, reduction = 'mean'
        jsdivergence = (F.kl_div(actual_normalized_log_probs, mix_normalized_log_probs.exp(), reduction = 'sum') +  F.kl_div(actual_normalized_log_probs, mix_normalized_log_probs.exp(), reduction = 'sum')) / 2.0
        print("js_div_standard_normal_empirical is {}".format(jsdivergence.item()))


        mse = ((self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols] - gt_proposal_deltas[fg_inds])**2).mean()
        predicted_variance = (self.pred_proposal_uncertain[fg_inds[:, None], gt_class_cols]).mean()

        loss_att_first_term = mse / predicted_variance

        storage = get_event_storage()
        storage.put_scalars(gt_mean=gt_mean,gt_variance= gt_variance,emp_mean=emp_mean,emp_variance=emp_var, mse = mse, predicted_variance = predicted_variance, loss_att_first_term = loss_att_first_term)
        storage.put_scalar("JSD-empirical/standard-normal", jsdivergence)

        return jsdivergence

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
        if self.loss_type is not 'smooth_l1':
            storage = get_event_storage()
            storage.put_scalar("smooth_l1_loss", loss_box_reg)

        return loss_box_reg


    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and regression loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_attenuation_final".
        """

        """
        'collect_residuals'
        'variance_loss',
        'smooth_l1',
        'loss_att',
        'loss_cal', 
        'mahalanobis_attenuation', 
        'wasserstein_over_chi_squared',
        'kl_div_chi_sq_closed_form_plus_smoothl1',
        'kl_div_standard_normal_closed_form_plus_smoothl1',
        'kl_div_chi_sq_empirical_plus_smoothl1',
        'kl_div_standard_normal_empirical_plus_smoothl1',
        'js_div_chi_sq_closed_form_plus_smoothl1',
        'js_div_standard_normal_closed_form_plus_smoothl1',
        'js_div_chi_sq_empirical_plus_smoothl1',
        'js_div_standard_normal_empirical_plus_smoothl1',
        'kl_div_chi_sq_closed_form',
        'kl_div_standard_normal_closed_form',
        'kl_div_chi_sq_empirical',
        'kl_div_standard_normal_empirical',
        'js_div_chi_sq_closed_form',
        'js_div_standard_normal_closed_form',
        'js_div_chi_sq_empirical',
        'js_div_standard_normal_empirical'
        'wasserstein_over_standard_normal'
        'wasserstein_over_standard_normal_plus_smoothl1',
        'kl_div_CLT_closed_form_gaussian'
        """

        w1, w2 = self.cfg.CUSTOM_OPTIONS.LOSS_WEIGHTS

        if self.loss_type == 'smooth_l1':
            loss_name = 'smooth_l1_loss'
            loss_reg = self.smooth_l1_loss()
        elif self.loss_type == 'collect_training_stats':
            loss_name = self.loss_type
            loss_reg = self.collect_training_stats()
        elif self.loss_type == 'collect_residuals':
            loss_name = self.loss_type
            loss_reg = self.collect_residuals()
        elif self.loss_type == 'loss_att':
            loss_name = 'loss_attenuation'
            loss_reg = self.loss_attenuation()
        elif self.loss_type == 'loss_calibration':
            loss_name = self.loss_type
            loss_reg = self.loss_calibration()
        elif self.loss_type == 'mahalanobis_attenuation':
            loss_name = 'mahalanobis_loss_attenuation'
            loss_reg = self.mahalanobis_loss_attenuation()
        elif self.loss_type == 'kl_div_chi_sq_closed_form_plus_smoothl1':
            loss_name = self.loss_type
            loss_reg = w1*self.kl_div_chi_sq_closed_form() + w2*self.smooth_l1_loss()
        elif self.loss_type == 'kl_div_standard_normal_closed_form_plus_smoothl1':
            loss_name = self.loss_type
            loss_reg = w1*self.kl_div_standard_normal_closed_form() + w2*self.smooth_l1_loss()
        elif self.loss_type == 'kl_div_chi_sq_empirical_plus_smoothl1':
            loss_name = self.loss_type
            loss_reg = w1*self.kl_div_chi_sq_empirical() + w2*self.smooth_l1_loss()
        elif self.loss_type == 'kl_div_standard_normal_empirical_plus_smoothl1':
            loss_name = self.loss_type
            loss_reg = w1*self.kl_div_standard_normal_empirical() + w2*self.smooth_l1_loss()
        elif self.loss_type == 'js_div_chi_sq_closed_form_plus_smoothl1':
            loss_name = self.loss_type
            loss_reg = w1*self.js_div_chi_sq_closed_form() + w2*self.smooth_l1_loss()
        elif self.loss_type == 'js_div_standard_normal_closed_form_plus_smoothl1':
            loss_name = self.loss_type
            loss_reg = w1*self.js_div_standard_normal_closed_form() + w2*self.smooth_l1_loss()
        elif self.loss_type == 'js_div_chi_sq_empirical_plus_smoothl1':
            loss_name = self.loss_type
            loss_reg = w1*self.js_div_chi_sq_empirical() + w2*self.smooth_l1_loss()
        elif self.loss_type == 'js_div_standard_normal_empirical_plus_smoothl1':
            loss_name = self.loss_type
            loss_reg = w1*self.js_div_standard_normal_empirical() + w2*self.smooth_l1_loss()
        elif self.loss_type == 'kl_div_CLT_closed_form_gaussian_plus_smoothl1':
            loss_name = self.loss_type
            loss_reg = w1*self.kl_div_CLT_closed_form_gaussian() + w2*self.smooth_l1_loss()

        elif self.loss_type == 'kl_div_chi_sq_closed_form':
            loss_name = self.loss_type
            loss_reg = self.kl_div_chi_sq_closed_form()
        elif self.loss_type == 'kl_div_standard_normal_closed':
            loss_name = self.loss_type
            loss_reg = self.kl_div_standard_normal_closed_form() 
        elif self.loss_type == 'kl_div_chi_sq_empirical':
            loss_name = self.loss_type
            loss_reg = self.kl_div_chi_sq_empirical() 
        elif self.loss_type == 'kl_div_standard_normal_empirical':
            loss_name = self.loss_type
            loss_reg = self.kl_div_standard_normal_empirical() 
        elif self.loss_type == 'js_div_chi_sq_closed_form':
            loss_name = self.loss_type
            loss_reg = self.js_div_chi_sq_closed_form()
        elif self.loss_type == 'js_div_standard_normal_closed_form':
            loss_name = self.loss_type
            loss_reg = self.js_div_standard_normal_closed_form() 
        elif self.loss_type == 'js_div_chi_sq_empirical':
            loss_name = self.loss_type
            loss_reg = self.js_div_chi_sq_empirical() 
        elif self.loss_type == 'js_div_standard_normal_empirical':
            loss_name = self.loss_type
            loss_reg = self.js_div_standard_normal_empirical()         
        elif self.loss_type == 'wasserstein_over_chi_squared':
            loss_name = self.loss_type
            loss_reg = self.wasserstein_over_chi_squared()
        elif self.loss_type == 'wasserstein_over_chi_squared_plus_smoothl1':
            loss_name = self.loss_type
            loss_reg = w1*self.wasserstein_over_chi_squared()  + w2*self.smooth_l1_loss()
        elif self.loss_type == 'wasserstein_over_standard_normal':
            loss_name = self.loss_type
            loss_reg = self.wasserstein_over_standard_normal() 
        elif self.loss_type == 'wasserstein_over_standard_normal_plus_smoothl1':
            loss_name = self.loss_type
            loss_reg = w1*self.wasserstein_over_standard_normal() + w2*self.smooth_l1_loss()
            # loss_reg = self.loss_attenuation()
        elif self.loss_type == 'variance_loss':
            loss_name = self.loss_type
            loss_reg = self.variance_loss()
        elif self.loss_type == 'kl_div_CLT_closed_form_gaussian':
            loss_name = self.loss_type
            loss_reg = self.kl_div_CLT_closed_form_gaussian()

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
                if cfg.CUSTOM_OPTIONS.NEW_UNCERTAINTY_HEAD:

                    self.bbox_uncertainty_pred = nn.Sequential(
                                                nn.Linear(input_size, int(3 * input_size / 4)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(3 * input_size / 4), int(input_size / 2)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(input_size / 2), num_bbox_reg_classes * box_dim),
                                              )
                    self.bbox_uncertainty_pred.apply(init_weights)

                else:
                    #### Adding uncertainty prediction(not the best way but a good start) ####
                    self.bbox_uncertainty_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
                    nn.init.normal_(self.bbox_uncertainty_pred.weight, std=0.001)
                    nn.init.constant_(self.bbox_uncertainty_pred.bias, 0)
                

            if cfg.CUSTOM_OPTIONS.LEARN_RC_SHARPNESS:
                self.sharpness = torch.nn.Parameter(data = torch.tensor(cfg.CUSTOM_OPTIONS.RICHARD_CURVE_SHARP).cuda(), requires_grad=True)
                # self.sharpness.requires_grad = True

            if cfg.CUSTOM_OPTIONS.TEMP_SCALE_ENABLED:
                self.temp_scale = torch.nn.Parameter(data = torch.tensor(1.0).cuda(), requires_grad=True)


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
        
        if self.cfg.CUSTOM_OPTIONS.LEARN_RC_SHARPNESS:
            print("richard's curve sharpness is: ", self.sharpness.item())
            return low + ((high - low) / (1 + torch.exp(-self.sharpness * x)))
        else:
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
                if self.cfg.CUSTOM_OPTIONS.TEMP_SCALE_ENABLED:
                    print("temp scale parameter value is: ", self.temp_scale.item())
                    proposal_delta_uncertainty = self.temp_scale**2 * proposal_delta_uncertainty
                return scores, proposal_deltas, proposal_delta_uncertainty
            else:
                return scores, proposal_deltas, None
        # import pdb; pdb.set_trace()
        return scores, proposal_deltas, None
        
