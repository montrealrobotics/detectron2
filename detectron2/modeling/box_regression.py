# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


__all__ = ["Box2BoxTransform", "Box2BoxTransformRotated", "Box2BoxXYXYTransform"]


class Box2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is parameterized
    by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
    """

    def __init__(self, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx1, dy1, dx2, dy2) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        assert torch.isfinite(deltas).all().item(), "Box regression deltas become infinite or NaN!"
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2
        return pred_boxes

class Box2BoxXYXYTransform(object):
    """
    The transformation is parameterized
    by 4 deltas: (dx1, dy1, dx2, dy2). The transformation shifts a box's corners 
    by the offset (dx * width, dy * height).

    Was used in https://arxiv.org/pdf/1809.08545.pdf
    """

    def __init__(self, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx1, dy1, dx2, dy2) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x1 = src_boxes[:, 0]
        src_ctr_y1 = src_boxes[:, 1]
        src_ctr_x2 = src_boxes[:, 2]
        src_ctr_y2 = src_boxes[:, 3]

        target_ctr_x1 = target_boxes[:, 0]
        target_ctr_y1 = target_boxes[:, 1]
        target_ctr_x2 = target_boxes[:, 2]
        target_ctr_y2 = target_boxes[:, 3]
        
        wx1, wy1, wx2, wy2 = self.weights

        dx1 = wx1 * (target_ctr_x1 - src_ctr_x1) / src_widths
        dx2 = wx2 * (target_ctr_x2 - src_ctr_x2) / src_widths
        dy1 = wy1 * (target_ctr_y1 - src_ctr_y1) / src_heights
        dy2 = wy2 * (target_ctr_y2 - src_ctr_y2) / src_heights
        

        deltas = torch.stack((dx1, dy1, dx2, dy2), dim=1)
        assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx1, dy1, dx2, dy2) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """

        assert torch.isfinite(deltas).all().item(), "Box regression deltas become infinite or NaN!"
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

       	wx1, wy1, wx2, wy2 = self.weights

        ## deltas
        dx1 = deltas[:, 0::4] / wx1
        dy1 = deltas[:, 1::4] / wy1
        dx2 = deltas[:, 2::4] / wx2
        dy2 = deltas[:, 3::4] / wy2

        x1_box = boxes[:, 0]
        y1_box = boxes[:, 1]
        x2_box = boxes[:, 2]
        y2_box = boxes[:, 3]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = dx1 * widths[:, None] + x1_box[:, None]  # x1
        pred_boxes[:, 1::4] = dy1 * heights[:, None] + y1_box[:, None]  # y1
        pred_boxes[:, 2::4] = dx2 * widths[:, None] + x2_box[:, None]  # x2  
        pred_boxes[:, 3::4] = dy2 * heights[:, None] + y2_box[:, None]  # y2
        return pred_boxes


    def apply_deltas_variance(self, deltas, boxes):

        assert torch.isfinite(deltas).all().item(), "Box regression deltas become infinite or NaN!"
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        wx1, wy1, wx2, wy2 = self.weights

        ## deltas
        dx1 = deltas[:, 0::4] / wx1**2
        dy1 = deltas[:, 1::4] / wy1**2
        dx2 = deltas[:, 2::4] / wx2**2
        dy2 = deltas[:, 3::4] / wy2**2

        # x1_box = boxes[:, 0]
        # y1_box = boxes[:, 1]
        # x2_box = boxes[:, 2]
        # y2_box = boxes[:, 3]

        pred_boxes_var = torch.zeros_like(deltas)

        ## What we are returning is standard deviation, which is square root of variance. 
        pred_boxes_var[:, 0::4] = (dx1 * (widths[:, None]**2))**0.5
        pred_boxes_var[:, 1::4] = (dy1 * (heights[:, None]**2))**0.5
        pred_boxes_var[:, 2::4] = (dx2 * (widths[:, None]**2))**0.5
        pred_boxes_var[:, 3::4] = (dy2 * (heights[:, None]**2))**0.5
        return pred_boxes_var


class Box2BoxTransformRotated(object):
    """
    The box-to-box transform defined in Rotated R-CNN. The transformation is parameterized
    by 5 deltas: (dx, dy, dw, dh, da). The transformation scales the box's width and height
    by exp(dw), exp(dh), shifts a box's center by the offset (dx * width, dy * height),
    and rotate a box's angle by da (radians).
    Note: angles of deltas are in radians while angles of boxes are in degrees.
    """

    def __init__(self, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):
        """
        Args:
            weights (5-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh, da) deltas. These are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh, da) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): Nx5 source boxes, e.g., object proposals
            target_boxes (Tensor): Nx5 target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_ctr_x, src_ctr_y, src_widths, src_heights, src_angles = torch.unbind(src_boxes, dim=1)

        target_ctr_x, target_ctr_y, target_widths, target_heights, target_angles = torch.unbind(
            target_boxes, dim=1
        )

        wx, wy, ww, wh, wa = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)
        # Angles of deltas are in radians while angles of boxes are in degrees.
        # the conversion to radians serve as a way to normalize the values
        da = target_angles - src_angles
        while len(torch.where(da < -180.0)[0]) > 0:
            da[torch.where(da < -180.0)] += 360.0
        while len(torch.where(da > 180.0)[0]) > 0:
            da[torch.where(da > 180.0)] -= 360.0
        da *= wa * math.pi / 180.0

        deltas = torch.stack((dx, dy, dw, dh, da), dim=1)
        assert (
            (src_widths > 0).all().item()
        ), "Input boxes to Box2BoxTransformRotated are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh, da) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, 5).
                deltas[i] represents box transformation for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 5)
        """
        assert deltas.shape[1] == 5 and boxes.shape[1] == 5
        assert torch.isfinite(deltas).all().item(), "Box regression deltas become infinite or NaN!"

        boxes = boxes.to(deltas.dtype)

        ctr_x, ctr_y, widths, heights, angles = torch.unbind(boxes, dim=1)
        wx, wy, ww, wh, wa = self.weights
        dx, dy, dw, dh, da = torch.unbind(deltas, dim=1)

        dx.div_(wx)
        dy.div_(wy)
        dw.div_(ww)
        dh.div_(wh)
        da.div_(wa)

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0] = dx * widths + ctr_x  # x_ctr
        pred_boxes[:, 1] = dy * heights + ctr_y  # y_ctr
        pred_boxes[:, 2] = torch.exp(dw) * widths  # width
        pred_boxes[:, 3] = torch.exp(dh) * heights  # height

        # Following original RRPN implementation,
        # angles of deltas are in radians while angles of boxes are in degrees.
        pred_angle = da * 180.0 / math.pi + angles

        while len(torch.where(pred_angle < -180.0)[0]) > 0:
            pred_angle[torch.where(pred_angle < -180.0)] += 360.0
        while len(torch.where(pred_angle > 180.0)[0]) > 0:
            pred_angle[torch.where(pred_angle > 180.0)] -= 360.0

        pred_boxes[:, 4] = pred_angle

        return pred_boxes
