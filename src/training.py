# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2019 Western Digital Corporation or its affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F

from config import NOOBJ_COEFF, COORD_COEFF, IGNORE_THRESH, ANCHORS, EPSILON

Tensor = torch.Tensor


def yolo_loss_fn(preds: Tensor, tgt: Tensor, tgt_len: Tensor, img_size: int, average=True):
    """Calculate the loss function given the predictions, the targets, the length of each target and the image size.
    Args:
        preds: (Tensor) the raw prediction tensor. Size is [B, N_PRED, NUM_ATTRIB],
                where B is the batch size;
                N_PRED is the total number of predictions, equivalent to 3*N_GRID*N_GRID for each scale;
                NUM_ATTRIB is the number of attributes, determined in config.py.
                coordinates is in format cxcywh and is local (raw).
                objectness is in logit.
                class score is in logit.
        tgt:   (Tensor) the tensor of ground truths (targets). Size is [B, N_tgt_max, NUM_ATTRIB].
                where N_tgt_max is the max number of targets in this batch.
                If a certain sample has targets fewer than N_tgt_max, zeros are filled at the tail.
        tgt_len: (Tensor) a 1D tensor showing the number of the targets for each sample. Size is [B, ].
        img_size: (int) the size of the training image.
        average: (bool) the flag of whether the loss is summed loss or average loss over the batch size.
    Return:
        the total loss
        """

    # generate the no-objectness mask. mask_noobj has size of [B, N_PRED]
    mask_noobj = noobj_mask_fn(preds, tgt)
    tgt_t_1d, idx_pred_obj = pre_process_targets(tgt, tgt_len, img_size)
    mask_noobj = noobj_mask_filter(mask_noobj, idx_pred_obj)

    # calculate the no-objectness loss
    pred_conf_logit = preds[..., 4]
    tgt_zero = torch.zeros(pred_conf_logit.size(), device=pred_conf_logit.device)
    # tgt_noobj = tgt_zero + (1 - mask_noobj) * 0.5
    pred_conf_logit = pred_conf_logit - (1 - mask_noobj) * 1e7
    noobj_loss = F.binary_cross_entropy_with_logits(pred_conf_logit, tgt_zero, reduction='sum')

    # select the predictions corresponding to the targets
    n_batch, n_pred, _ = preds.size()
    preds_1d = preds.view(n_batch * n_pred, -1)
    preds_obj = preds_1d.index_select(0, idx_pred_obj)

    # calculate the coordinate loss
    coord_loss = F.mse_loss(preds_obj[..., :4], tgt_t_1d[..., :4], reduction='sum')
    # assert not torch.isnan(coord_loss)

    # calculate the objectness loss
    pred_conf_obj_logit = preds_obj[..., 4]
    tgt_one = torch.ones(pred_conf_obj_logit.size(), device=pred_conf_obj_logit.device)
    obj_loss = F.binary_cross_entropy_with_logits(pred_conf_obj_logit, tgt_one, reduction='sum')

    # calculate the classification loss
    class_loss = F.binary_cross_entropy_with_logits(preds_obj[..., 5:], tgt_t_1d[..., 5:], reduction='sum')

    # total loss
    total_loss = noobj_loss * NOOBJ_COEFF + obj_loss + class_loss + coord_loss * COORD_COEFF

    if average:
        total_loss = total_loss / n_batch

    return total_loss, coord_loss, obj_loss, noobj_loss, class_loss


def noobj_mask_fn(pred: Tensor, target: Tensor):
    """pred is a 3D tensor with shape
    (num_batch, NUM_ANCHORS_PER_SCALE*num_grid*num_grid, NUM_ATTRIB). The raw data has been converted.
    target is a 3D tensor with shape
    (num_batch, max_num_object, NUM_ATTRIB).
     The max_num_objects depend on the sample which has max num_objects in this minibatch"""
    num_batch, num_pred, num_attrib = pred.size()
    assert num_batch == target.size(0)
    ious = iou_batch(pred[..., :4], target[..., :4], center=True) #in cxcywh format
    # for each pred bbox, find the target box which overlaps with it (without zero centered) most, and the iou value.
    max_ious, max_ious_idx = torch.max(ious, dim=2)
    noobj_indicator = torch.where((max_ious - IGNORE_THRESH) > 0, torch.zeros_like(max_ious), torch.ones_like(max_ious))
    return noobj_indicator


def noobj_mask_filter(mask_noobj: Tensor, idx_obj_1d: Tensor):
    n_batch, n_pred = mask_noobj.size()
    mask_noobj = mask_noobj.view(-1)
    filter_ = torch.zeros(mask_noobj.size(), device=mask_noobj.device)
    mask_noobj.scatter_(0, idx_obj_1d, filter_)
    mask_noobj = mask_noobj.view(n_batch, -1)
    return mask_noobj


def pre_process_targets(tgt: Tensor, tgt_len, img_size):
    """get the index of the predictions corresponding to the targets;
    and put targets from different sample into one dimension (flatten), getting rid of the tails;
    and convert coordinates to local.
    Args:
        tgt: (tensor) the tensor of ground truths (targets). Size is [B, N_tgt_max, NUM_ATTRIB].
                    where B is the batch size;
                    N_tgt_max is the max number of targets in this batch;
                    NUM_ATTRIB is the number of attributes, determined in config.py.
                    coordinates is in format cxcywh and is global.
                    If a certain sample has targets fewer than N_tgt_max, zeros are filled at the tail.
        tgt_len: (Tensor) a 1D tensor showing the number of the targets for each sample. Size is [B, ].
        img_size: (int) the size of the training image.
    :return
        tgt_t_flat: (tensor) the flattened and local target. Size is [N_tgt_total, NUM_ATTRIB],
                            where N_tgt_total is the total number of targets in this batch.
        idx_obj_1d: (tensor) the tensor of the indices of the predictions corresponding to the targets.
                            The size is [N_tgt_total, ]. Note the indices have been added the batch number,
                            therefore when the predictions are flattened, the indices can directly find the prediction.
    """
    # find the anchor box which has max IOU (zero centered) with the targets
    wh_anchor = torch.tensor(ANCHORS).to(tgt.device).float()
    n_anchor = wh_anchor.size(0)
    xy_anchor = torch.zeros((n_anchor, 2), device=tgt.device)
    bbox_anchor = torch.cat((xy_anchor, wh_anchor), dim=1)
    bbox_anchor.unsqueeze_(0)
    iou_anchor_tgt = iou_batch(bbox_anchor, tgt[..., :4], zero_center=True)
    _, idx_anchor = torch.max(iou_anchor_tgt, dim=1)

    # find the corresponding prediction's index for the anchor box with the max IOU
    strides_selection = [8, 16, 32]
    scale = idx_anchor // 3
    idx_anchor_by_scale = idx_anchor - scale * 3
    stride = 8 * 2 ** scale
    grid_x = (tgt[..., 0] // stride.float()).long()
    grid_y = (tgt[..., 1] // stride.float()).long()
    n_grid = img_size // stride
    large_scale_mask = (scale <= 1).long()
    med_scale_mask = (scale <= 0).long()
    idx_obj = \
        large_scale_mask * (img_size // strides_selection[2]) ** 2 * 3 + \
        med_scale_mask * (img_size // strides_selection[1]) ** 2 * 3 + \
        n_grid ** 2 * idx_anchor_by_scale + n_grid * grid_y + grid_x

    # calculate t_x and t_y
    t_x = (tgt[..., 0] / stride.float() - grid_x.float()).clamp(EPSILON, 1 - EPSILON)
    t_x = torch.log(t_x / (1. - t_x))   #inverse of sigmoid
    t_y = (tgt[..., 1] / stride.float() - grid_y.float()).clamp(EPSILON, 1 - EPSILON)
    t_y = torch.log(t_y / (1. - t_y))    # inverse of sigmoid

    # calculate t_w and t_h
    w_anchor = wh_anchor[..., 0]
    h_anchor = wh_anchor[..., 1]
    w_anchor = torch.index_select(w_anchor, 0, idx_anchor.view(-1)).view(idx_anchor.size())
    h_anchor = torch.index_select(h_anchor, 0, idx_anchor.view(-1)).view(idx_anchor.size())
    t_w = torch.log((tgt[..., 2] / w_anchor).clamp(min=EPSILON))
    t_h = torch.log((tgt[..., 3] / h_anchor).clamp(min=EPSILON))

    # the raw target tensor
    tgt_t = tgt.clone().detach()

    tgt_t[..., 0] = t_x
    tgt_t[..., 1] = t_y
    tgt_t[..., 2] = t_w
    tgt_t[..., 3] = t_h

    # aggregate processed targets and the corresponding prediction index from different batches in to one dimension
    n_batch = tgt.size(0)
    n_pred = sum([(img_size // s) ** 2 for s in strides_selection]) * 3

    idx_obj_1d = []
    tgt_t_flat = []

    for i_batch in range(n_batch):
        v = idx_obj[i_batch]
        t = tgt_t[i_batch]
        l = tgt_len[i_batch]
        idx_obj_1d.append(v[:l] + i_batch * n_pred)
        tgt_t_flat.append(t[:l])

    idx_obj_1d = torch.cat(idx_obj_1d)
    tgt_t_flat = torch.cat(tgt_t_flat)

    return tgt_t_flat, idx_obj_1d


def iou_batch(bboxes1: Tensor, bboxes2: Tensor, center=False, zero_center=False):
    """Calculate the IOUs between bboxes1 and bboxes2.
    :param
      bboxes1: (Tensor) A 3D tensor representing first group of bboxes.
        The dimension is (B, N1, 4). B is the number of samples in the batch.
        N1 is the number of bboxes in each sample.
        The third dimension represent the bbox, with coordinate (x, y, w, h) or (cx, cy, w, h).
      bboxes2: (Tensor) A 3D tensor representing second group of bboxes.
        The dimension is (B, N2, 4). It is similar to bboxes1, the only difference is N2.
        N1 is the number of bboxes in each sample.
      center: (bool). Whether the bboxes are in format (cx, cy, w, h).
      zero_center: (bool). Whether to align two bboxes so their center is aligned.
    :return
      iou_: (Tensor) A 3D tensor representing the IOUs.
        The dimension is (B, N1, N2)."""
    x1 = bboxes1[..., 0]
    y1 = bboxes1[..., 1]
    w1 = bboxes1[..., 2]
    h1 = bboxes1[..., 3]

    x2 = bboxes2[..., 0]
    y2 = bboxes2[..., 1]
    w2 = bboxes2[..., 2]
    h2 = bboxes2[..., 3]

    area1 = w1 * h1
    area2 = w2 * h2

    if zero_center:
        w1.unsqueeze_(2)
        w2.unsqueeze_(1)
        h1.unsqueeze_(2)
        h2.unsqueeze_(1)
        w_intersect = torch.min(w1, w2).clamp(min=0)
        h_intersect = torch.min(h1, h2).clamp(min=0)
    else:
        if center:
            x1 = x1 - w1 / 2
            y1 = y1 - h1 / 2
            x2 = x2 - w2 / 2
            y2 = y2 - h2 / 2
        right1 = (x1 + w1).unsqueeze(2)
        right2 = (x2 + w2).unsqueeze(1)
        top1 = (y1 + h1).unsqueeze(2)
        top2 = (y2 + h2).unsqueeze(1)
        left1 = x1.unsqueeze(2)
        left2 = x2.unsqueeze(1)
        bottom1 = y1.unsqueeze(2)
        bottom2 = y2.unsqueeze(1)
        w_intersect = (torch.min(right1, right2) - torch.max(left1, left2)).clamp(min=0)
        h_intersect = (torch.min(top1, top2) - torch.max(bottom1, bottom2)).clamp(min=0)
    area_intersect = h_intersect * w_intersect

    iou_ = area_intersect / (area1.unsqueeze(2) + area2.unsqueeze(1) - area_intersect + EPSILON)

    return iou_
