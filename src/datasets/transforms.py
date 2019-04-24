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

import random
import torch
import numbers

import numpy as np
from PIL import Image
from torchvision import transforms as tv_tf
from torchvision.transforms import functional as TF

from config import EPSILON


# def default_transform_fn_old(padding, img_size):
#     return tv_tf.Compose([tv_tf.Pad(padding, fill=(127, 127, 127)),
#                           tv_tf.Resize(img_size),
#                           tv_tf.ToTensor()])


def default_transform_fn(img_size):
    return ComposeWithLabel([PadToSquareWithLabel(fill=(127, 127, 127)),
                             ResizeWithLabel(img_size),
                             tv_tf.ToTensor()])


def random_transform_fn(img_size):
    return ComposeWithLabel([RandomHorizontalFlipWithLabel(),
                             RandomResizedCropWithLabel(img_size, scale=(0.9, 1.1)),
                             RandomAffineWithLabel(degrees=5, shear=10),
                             RandomAdjustImage(),
                             ClampLabel(),
                             tv_tf.ToTensor()])


class RandomResizedCropWithLabel(tv_tf.RandomResizedCrop):

    def __call__(self, img, label=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img_tf = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        if label is not None:
            label[..., 0] -= j
            label[..., 1] -= i
            h_ratio = self.size[1] / h
            w_ratio = self.size[0] / w
            label[..., 0] *= w_ratio
            label[..., 1] *= h_ratio
            label[..., 2] *= w_ratio
            label[..., 3] *= h_ratio
            return img_tf, label
        else:
            return img_tf


class ClampLabel(object):

    def __init__(self, min_w=4, min_h=4, min_area=None):
        self.min_w = min_w
        self.min_h = min_h
        self.min_area = min_area if min_area is not None else min_w * min_h

    def __call__(self, img, label):
        w, h = img.size
        label[..., 2] += label[..., 0]
        label[..., 3] += label[..., 1]
        label = label[label[..., 0] < w - EPSILON]
        label = label[label[..., 1] < h - EPSILON]
        label = label[label[..., 2] > EPSILON]
        label = label[label[..., 3] > EPSILON]
        label[..., 0] = torch.clamp(label[..., 0], min=0)
        label[..., 1] = torch.clamp(label[..., 1], min=0)
        label[..., 2] = torch.clamp(label[..., 2], max=w)
        label[..., 3] = torch.clamp(label[..., 3], max=h)
        label[..., 2] -= label[..., 0]
        label[..., 3] -= label[..., 1]
        label = label[label[..., 2] >= self.min_w]
        label = label[label[..., 3] >= self.min_h]
        label = label[label[..., 2] * label[..., 3] >= self.min_area]
        return img, label

    def __repr__(self):
        pass


class RandomHorizontalFlipWithLabel(tv_tf.RandomHorizontalFlip):

    def __call__(self, img: Image.Image, label=None):
        """
        Args:
            img (PIL Image): Image to be flipped.
            label (Torch Tensor): bounding boxes of the image
        Returns:
            PIL Image: Randomly flipped image.
        """
        if label is None:
            return super(RandomHorizontalFlipWithLabel, self).__call__(img)
        if random.random() < self.p:
            label[..., 0] = img.width - label[..., 0] - label[..., 2]
            img = TF.hflip(img)
        return img, label


class RandomRotationWithLabel(tv_tf.RandomRotation):

    def __call__(self, img: Image.Image, label=None):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        if label is None:
            return super(RandomRotationWithLabel, self).__call__(img)
        angle = self.get_params(self.degrees)

        return TF.rotate(img, angle, self.resample, self.expand, self.center)


class RandomAffineWithLabel(tv_tf.RandomAffine):

    def __call__(self, img: Image.Image, label=None):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        if label is None:
            return super(RandomAffineWithLabel, self).__call__(img)
        rot_angle, translate, scale, shear = \
            self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        img = TF.affine(img, rot_angle, translate, scale, shear, resample=self.resample, fillcolor=self.fillcolor)
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
        affine_transform_matrix = _get_affine_matrix(center, rot_angle, translate, scale, shear)
        label = _affine_transform_label(label, affine_transform_matrix)
        return img, label


class ComposeWithLabel(tv_tf.Compose):

    def __call__(self, img, label=None):
        import inspect
        for t in self.transforms:
            num_param = len(inspect.signature(t).parameters)
            if num_param == 2:
                img, label = t(img, label)
            elif num_param == 1:
                img = t(img)
        return img, label


class RandomAdjustImage(object):

    def __init__(self,
                 brightness=(0.9, 1.1),
                 contrast=(0.5, 1.5),
                 gamma=(0.5, 1.5),
                 hue=(0, 0),
                 saturation=(0.5, 1.5)):
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.hue = hue
        self.saturation = saturation

    @staticmethod
    def get_param(ranges):
        samples = []
        for r in ranges:
            samples.append(random.uniform(*r))
        return samples

    def __call__(self, img):
        b, c, g, h, s = self.get_param([self.brightness, self.contrast, self.gamma, self.hue, self.saturation])
        img = TF.adjust_brightness(img, b)
        img = TF.adjust_contrast(img, c)
        img = TF.adjust_gamma(img, g)
        img = TF.adjust_hue(img, h)
        img = TF.adjust_saturation(img, s)
        return img


class PadToSquareWithLabel(object):
    """Pad to square the given PIL Image with label.
    Args:
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value at the edge of the image
            - reflect: pads with reflection of image without repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def _get_padding(w, h):
        """Generate the size of the padding given the size of the image,
        such that the padded image will be square.
        Args:
            h (int): the height of the image.
            w (int): the width of the image.
        Return:
            A tuple of size 4 indicating the size of the padding in 4 directions:
            left, top, right, bottom. This is to match torchvision.transforms.Pad's parameters.
            For details, see:
                https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Pad
            """
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        return (0, pad1, 0, pad2) if h <= w else (pad1, 0, pad2, 0)

    def __call__(self, img, label=None):
        w, h = img.size
        padding = self._get_padding(w, h)
        img = TF.pad(img, padding, self.fill, self.padding_mode)
        if label is None:
            return img, label
        label[..., 0] += padding[0]
        label[..., 1] += padding[1]
        return img, label


class ResizeWithLabel(tv_tf.Resize):

    def __init__(self, size, interpolation=Image.BILINEAR):
        super(ResizeWithLabel, self).__init__(size, interpolation)

    def __call__(self, img, label=None):
        w_old, h_old = img.size
        img = super(ResizeWithLabel, self).__call__(img)
        w_new, h_new = img.size
        if label is None:
            return img, label
        scale_w = w_new / w_old
        scale_h = h_new / h_old
        label[..., 0] *= scale_w
        label[..., 1] *= scale_h
        label[..., 2] *= scale_w
        label[..., 3] *= scale_h
        return img, label


def _get_affine_matrix(center, angle, translate, scale, shear):
    """Helper method to compute matrix for affine transformation

    We need compute affine transformation matrix: M = T * C * RSS * C^-1
    where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
          C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
          RSS is rotation with scale and shear matrix
          RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
                                 [ sin(a)*scale    cos(a + shear)*scale     0]
                                 [     0                  0          1]"""
    angle = np.deg2rad(angle)
    shear = np.deg2rad(shear)
    C = np.matrix([[1., 0., center[0]],
                   [0., 1., center[1]],
                   [0., 0.,        1.]])
    T = np.matrix([[1., 0., translate[0]],
                   [0., 1., translate[1]],
                   [0., 0.,           1.]])
    RSS = np.matrix([[np.cos(angle), -np.sin(angle + shear),       0.],
                     [np.sin(angle),  np.cos(angle + shear),       0.],
                     [           0.,                     0., 1./scale]]) * scale
    M = T * C * RSS * np.linalg.inv(C)
    return M


def _affine_transform_label(label, affine_matrix):
    xywh = np.matrix(label[..., :4].numpy())
    xy_lt = xywh[:, :2].copy()
    xy_rb = xy_lt + xywh[:, 2:4].copy()
    xy_rt = xywh[:, :2].copy()
    xy_rt[:, 0] = xy_rb[:, 0].copy()
    xy_lb = xywh[:, :2].copy()
    xy_lb[:, 1] = xy_rb[:, 1].copy()
    rotation = affine_matrix[:2, :2]
    translation = affine_matrix[:2, 2]
    xy_lt = xy_lt.dot(rotation.T) + translation.T
    xy_rb = xy_rb.dot(rotation.T) + translation.T
    xy_rt = xy_rt.dot(rotation.T) + translation.T
    xy_lb = xy_lb.dot(rotation.T) + translation.T
    x1 = np.minimum(xy_lt[:, 0], xy_lb[:, 0])
    y1 = np.minimum(xy_lt[:, 1], xy_rt[:, 1])
    x2 = np.maximum(xy_rt[:, 0], xy_rb[:, 0])
    y2 = np.maximum(xy_lb[:, 1], xy_rb[:, 1])
    xywh = np.concatenate((x1, y1, x2 - x1, y2 - y1), axis=1)
    label[..., :4] = torch.from_numpy(xywh)
    return label
