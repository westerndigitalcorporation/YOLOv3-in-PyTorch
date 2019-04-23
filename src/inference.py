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

import logging
import torch

from config import NUM_CLASSES, EPSILON


def post_process(results_raw, nms, conf_thres, nms_thres):
    results = []
    for idx, result_raw in enumerate(results_raw):
        bboxes = result_raw[..., :4]
        scores = result_raw[..., 4]
        classes_one_hot = result_raw[..., 5:]
        classes = torch.argmax(classes_one_hot, dim=1)
        if nms:
            bboxes, scores, classes = \
                non_max_suppression(bboxes, scores, classes,
                                    num_classes=NUM_CLASSES,
                                    center=True,
                                    conf_thres=conf_thres,
                                    nms_thres=nms_thres)
        result = torch.cat((bboxes, scores.view((-1, 1)), classes.view((-1, 1)).float()), dim=1)
        results.append(result)
        logging.debug("The dimension of the result after nms is {} for idx {}".format(result.size(), idx))
    return results


def non_max_suppression(bboxes, scores, classes, num_classes, conf_thres=0.8, nms_thres=0.5, center=False):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        bboxes: (tensor) The location predictions for the img, Shape: [num_priors,4].
        scores: (tensor) The class prediction scores for the img, Shape:[num_priors].
        classes: (tensor) The label (non-one-hot) representation of the classes of the objects,
          Shape: [num_priors].
        num_classes: (int) The number of all the classes.
        conf_thres: (float) Threshold where all the detections below this value will be ignored.
        nms_thres: (float) The overlap thresh for suppressing unnecessary boxes.
        center: (boolean) Whether the bboxes format is cxcywh or xywh.
    Return:
        The indices of the kept boxes with respect to num_priors, and they are always in xywh format.
    """

    # make sure bboxes and scores have the same 0th dimension
    assert bboxes.shape[0] == scores.shape[0] == classes.shape[0]
    num_prior = bboxes.shape[0]

    # if no objects, return raw result
    if num_prior == 0:
        return bboxes, scores, classes

    # threshold out low confidence detection

    if conf_thres > 0:

        conf_index = torch.nonzero(torch.ge(scores, conf_thres)).squeeze()

        bboxes = bboxes.index_select(0, conf_index)
        scores = scores.index_select(0, conf_index)
        classes = classes.index_select(0, conf_index)

    # if there are multiple classes, divide them into groups
    grouped_indices = group_same_class_object(classes, one_hot=False, num_classes=num_classes)
    selected_indices_final = []

    for class_id, member_idx in enumerate(grouped_indices):
        member_idx_tensor = bboxes.new_tensor(member_idx, dtype=torch.long)
        bboxes_one_class = bboxes.index_select(dim=0, index=member_idx_tensor)
        scores_one_class = scores.index_select(dim=0, index=member_idx_tensor)
        scores_one_class, sorted_indices = torch.sort(scores_one_class, descending=False)

        selected_indices = []

        while sorted_indices.size(0) != 0:
            picked_index = sorted_indices[-1]
            selected_indices.append(picked_index)
            picked_bbox = bboxes_one_class[picked_index]
            ious = iou_one_to_many(picked_bbox, bboxes_one_class[sorted_indices[:-1]], center=center)
            under_indices = torch.nonzero(ious <= nms_thres).squeeze()
            sorted_indices = sorted_indices.index_select(dim=0, index=under_indices)

        selected_indices_final.extend([member_idx[i] for i in selected_indices])

    selected_indices_final = bboxes.new_tensor(selected_indices_final, dtype=torch.long)
    bboxes_result = bboxes.index_select(dim=0, index=selected_indices_final)
    scores_result = scores.index_select(dim=0, index=selected_indices_final)
    classes_result = classes.index_select(dim=0, index=selected_indices_final)

    return bboxes_result, scores_result, classes_result


def group_same_class_object(obj_classes, one_hot=True, num_classes=-1):
    """
    Given a list of class results, group the object with the same class into a list.
    Returns a list with the length of num_classes, where each bucket has the objects with the same class.
    :param
        obj_classes: (torch.tensor) The representation of classes of object.
         It can be either one-hot or label (non-one-hot).
         If it is one-hot, the shape should be: [num_objects, num_classes]
         If it is label (non-non-hot), the shape should be: [num_objects, ]
        one_hot: (bool) A flag telling the function whether obj_classes is one-hot representation.
        num_classes: (int) The max number of classes if obj_classes is represented as non-one-hot format.
    :return:
        a list of of a list, where for the i-th list,
        the elements in such list represent the indices of the objects in class i.
    """
    if one_hot:
        num_classes = obj_classes.shape[-1]
    else:
        assert num_classes != -1
    grouped_index = [[] for _ in range(num_classes)]
    if one_hot:
        for idx, class_one_hot in enumerate(obj_classes):
            grouped_index[torch.argmax(class_one_hot)].append(idx)
    else:
        for idx, obj_class_ in enumerate(obj_classes):
            grouped_index[obj_class_].append(idx)
    return grouped_index


def iou(bbox1, bbox2, center=False):
    """Calculate IOU for two bboxes. If center is false, then they should all in xywh format.
    Else, they should all be in cxcywh format"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    if center:
        x1 = x1 - w1 / 2
        y1 = y1 - h1 / 2
        x2 = x2 - w2 / 2
        y2 = y2 - h2 / 2
    area1 = w1 * h1
    area2 = w2 * h2
    right1 = x1 + w1
    right2 = x2 + w2
    bottom1 = y1 + h1
    bottom2 = y2 + h2
    w_intersect = (torch.min(right1, right2) - torch.max(x1, x2)).clamp(min=0)
    h_intersect = (torch.min(bottom1, bottom2) - torch.max(y1, y2)).clamp(min=0)
    area_intersect = w_intersect * h_intersect
    iou_ = area_intersect / (area1 + area2 - area_intersect + EPSILON) #add epsilon to avoid NaN
    return iou_


def iou_one_to_many(bbox1, bboxes2, center=False):
    """Calculate IOU for one bbox with another group of bboxes.
    If center is false, then they should all in xywh format.
    Else, they should all be in cxcywh format"""
    x1, y1, w1, h1 = bbox1
    x2 = bboxes2[..., 0]
    y2 = bboxes2[..., 1]
    w2 = bboxes2[..., 2]
    h2 = bboxes2[..., 3]
    if center:
        x1 = x1 - w1 / 2
        y1 = y1 - h1 / 2
        x2 = x2 - w2 / 2
        y2 = y2 - h2 / 2
    area1 = w1 * h1
    area2 = w2 * h2
    right1 = x1 + w1
    right2 = x2 + w2
    bottom1 = y1 + h1
    bottom2 = y2 + h2
    w_intersect = (torch.min(right1, right2) - torch.max(x1, x2)).clamp(min=0)
    h_intersect = (torch.min(bottom1, bottom2) - torch.max(y1, y2)).clamp(min=0)
    area_intersect = w_intersect * h_intersect
    iou_ = area_intersect / (area1 + area2 - area_intersect + EPSILON) #add epsilon to avoid NaN
    return iou_


def argsort(t, reverse=False):
    """Given a list, sort the list and return the original indices of the sorted list."""
    return sorted(range(len(t)), key=t.__getitem__, reverse=reverse)
