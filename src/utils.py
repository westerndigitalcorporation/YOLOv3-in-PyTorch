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
import torch.nn as nn

from config import MISSING_IDS

from PIL import ImageDraw, ImageFont
from torchvision.transforms import ToPILImage


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


def draw_result(img, boxes, show=False, class_names = None):
    if isinstance(img, torch.Tensor):
        transform = ToPILImage()
        img = transform(img)
    draw = ImageDraw.ImageDraw(img)
    show_class = (boxes.size(1) >= 6)
    if show_class:
        assert isinstance(class_names, list)
    for box in boxes:
        x, y, w, h = box[:4]
        x2 = x + w
        y2 = y + h
        draw.rectangle([x, y, x2, y2], outline='white', width=3)
        if show_class:
            class_id = int(box[5])
            class_name = class_names[class_id]
            font_size = 20
            class_font = ImageFont.truetype("../fonts/Roboto-Regular.ttf", font_size)
            text_size = draw.textsize(class_name, font=class_font)
            draw.rectangle([x, y-text_size[1], x + text_size[0], y], fill='white')
            draw.text([x, y-font_size], class_name, font=class_font, fill='black')
    if show:
        img.show()
    return img


