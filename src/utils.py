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

import numpy as np
import torch
import torch.nn as nn

from config import MISSING_IDS, NUM_CLASSES_COCO


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def init_conv_layer_randomly(m):
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    if m.bias is not None:
        torch.nn.init.constant_(m.bias.data, 0.0)


def init_bn_layer_randomly(m):
    torch.nn.init.constant_(m.weight, 1.0)
    torch.nn.init.constant_(m.bias, 0.0)


def init_layer_randomly(m):
    if isinstance(m, nn.Conv2d):
        init_conv_layer_randomly(m)
    elif isinstance(m, nn.BatchNorm2d):
        init_bn_layer_randomly(m)
    else:
        pass


def untransform_bboxes(bboxes, scale, padding):
    """transform the bounding box from the scaled image back to the unscaled image."""
    x = bboxes[..., 0]
    y = bboxes[..., 1]
    w = bboxes[..., 2]
    h = bboxes[..., 3]
    # x, y, w, h = bbs
    x /= scale
    y /= scale
    w /= scale
    h /= scale
    x -= padding[0]
    y -= padding[1]
    return bboxes


def transform_bboxes(bb, scale, padding):
    """transform the bounding box from the raw image  to the padded-then-scaled image."""
    x, y, w, h = bb
    x += padding[0]
    y += padding[1]
    x *= scale
    y *= scale
    w *= scale
    h *= scale

    return x, y, w, h


def coco_category_to_one_hot(category_id, dtype="uint"):
    """ convert from a category_id to one-hot vector, considering there are missing IDs in coco dataset."""
    new_id = delete_coco_empty_category(category_id)
    return category_to_one_hot(new_id, NUM_CLASSES_COCO, dtype)


def category_to_one_hot(category_id, num_classes, dtype="uint"):
    """ convert from a category_id to one-hot vector """
    return torch.from_numpy(np.eye(num_classes, dtype=dtype)[category_id])


def delete_coco_empty_category(old_id):
    """The COCO dataset has 91 categories but 11 of them are empty.
    This function will convert the 80 existing classes into range [0-79].
    Note the COCO original class index starts from 1.
    The converted index starts from 0.
    Args:
        old_id (int): The category ID from COCO dataset.
    Return:
        new_id (int): The new ID after empty categories are removed. """
    starting_idx = 1
    new_id = old_id - starting_idx
    for missing_id in MISSING_IDS:
        if old_id > missing_id:
            new_id -= 1
        elif old_id == missing_id:
            raise KeyError("illegal category ID in coco dataset! ID # is {}".format(old_id))
        else:
            break
    return new_id


def add_coco_empty_category(old_id):
    """The reverse of delete_coco_empty_category."""
    starting_idx = 1
    new_id = old_id + starting_idx
    for missing_id in MISSING_IDS:
        if new_id >= missing_id:
            new_id += 1
        else:
            break
    return new_id


def cxcywh_to_xywh(bbox):
    bbox[..., 0] -= bbox[..., 2] / 2
    bbox[..., 1] -= bbox[..., 3] / 2
    return bbox


def xywh_to_cxcywh(bbox):
    bbox[..., 0] += bbox[..., 2] / 2
    bbox[..., 1] += bbox[..., 3] / 2
    return bbox


def draw_result(img, boxes):
    img = np.array(img.numpy().transpose(1, 2, 0))
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        bbox = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='b',
                                 facecolor='none')
        ax.add_patch(bbox)
    plt.show()


