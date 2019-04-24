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
import torch.nn as nn
import torch.nn.functional as F

from config import ANCHORS, NUM_ANCHORS_PER_SCALE, NUM_CLASSES, NUM_ATTRIB, LAST_LAYER_DIM

Tensor = torch.Tensor


class ConvLayer(nn.Module):
    """Basic 'conv' layer, including:
     A Conv2D layer with desired channels and kernel size,
     A batch-norm layer,
     and A leakyReLu layer with neg_slope of 0.1.
     (Didn't find too much resource what neg_slope really is.
     By looking at the darknet source code, it is confirmed the neg_slope=0.1.
     Ref: https://github.com/pjreddie/darknet/blob/master/src/activations.h)
     Please note here we distinguish between Conv2D layer and Conv layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, lrelu_neg_slope=0.1):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_neg_slope)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)

        return out


class ResBlock(nn.Module):
    """The basic residual block used in YoloV3.
    Each ResBlock consists of two ConvLayers and the input is added to the final output.
    In YoloV3 paper, the first convLayer has half of the number of the filters as much as the second convLayer.
    The first convLayer has filter size of 1x1 and the second one has the filter size of 3x3.
    """

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        assert in_channels % 2 == 0  # ensure the in_channels is an even number.
        half_in_channels = in_channels // 2
        self.conv1 = ConvLayer(in_channels, half_in_channels, 1)
        self.conv2 = ConvLayer(half_in_channels, in_channels, 3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual

        return out


def make_conv_and_res_block(in_channels, out_channels, res_repeat):
    """In Darknet 53 backbone, there is usually one Conv Layer followed by some ResBlock.
    This function will make that.
    The Conv layers always have 3x3 filters with stride=2.
    The number of the filters in Conv layer is the same as the out channels of the ResBlock"""
    model = nn.Sequential()
    model.add_module('conv', ConvLayer(in_channels, out_channels, 3, stride=2))
    for idx in range(res_repeat):
        model.add_module('res{}'.format(idx), ResBlock(out_channels))
    return model


class YoloLayer(nn.Module):

    def __init__(self, scale, stride):
        super(YoloLayer, self).__init__()
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            idx = None
        self.anchors = torch.tensor([ANCHORS[i] for i in idx])
        self.stride = stride

    def forward(self, x):
        num_batch = x.size(0)
        num_grid = x.size(2)

        if self.training:
            output_raw = x.view(num_batch,
                                NUM_ANCHORS_PER_SCALE,
                                NUM_ATTRIB,
                                num_grid,
                                num_grid).permute(0, 1, 3, 4, 2).contiguous().view(num_batch, -1, NUM_ATTRIB)
            return output_raw
        else:
            prediction_raw = x.view(num_batch,
                                    NUM_ANCHORS_PER_SCALE,
                                    NUM_ATTRIB,
                                    num_grid,
                                    num_grid).permute(0, 1, 3, 4, 2).contiguous()

            self.anchors = self.anchors.to(x.device).float()
            # Calculate offsets for each grid
            grid_tensor = torch.arange(num_grid, dtype=torch.float, device=x.device).repeat(num_grid, 1)
            grid_x = grid_tensor.view([1, 1, num_grid, num_grid])
            grid_y = grid_tensor.t().view([1, 1, num_grid, num_grid])
            anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
            anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))

            # Get outputs
            x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * self.stride # Center x
            y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * self.stride  # Center y
            w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w  # Width
            h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h  # Height
            bbox_pred = torch.stack((x_center_pred, y_center_pred, w_pred, h_pred), dim=4).view((num_batch, -1, 4)) #cxcywh
            conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)  # Conf
            cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, NUM_CLASSES)  # Cls pred one-hot.

            output = torch.cat((bbox_pred, conf_pred, cls_pred), -1)
            return output


class DetectionBlock(nn.Module):
    """The DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
    1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n,
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)
    out_channels = n
    """

    def __init__(self, in_channels, out_channels, scale, stride):
        super(DetectionBlock, self).__init__()
        assert out_channels % 2 == 0  #assert out_channels is an even number
        half_out_channels = out_channels // 2
        self.conv1 = ConvLayer(in_channels, half_out_channels, 1)
        self.conv2 = ConvLayer(half_out_channels, out_channels, 3)
        self.conv3 = ConvLayer(out_channels, half_out_channels, 1)
        self.conv4 = ConvLayer(half_out_channels, out_channels, 3)
        self.conv5 = ConvLayer(out_channels, half_out_channels, 1)
        self.conv6 = ConvLayer(half_out_channels, out_channels, 3)
        self.conv7 = nn.Conv2d(out_channels, LAST_LAYER_DIM, 1, bias=True)
        self.yolo = YoloLayer(scale, stride)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        self.branch = self.conv5(tmp)
        tmp = self.conv6(self.branch)
        tmp = self.conv7(tmp)
        out = self.yolo(tmp)

        return out


class DarkNet53BackBone(nn.Module):

    def __init__(self):
        super(DarkNet53BackBone, self).__init__()
        self.conv1 = ConvLayer(3, 32, 3)
        self.cr_block1 = make_conv_and_res_block(32, 64, 1)
        self.cr_block2 = make_conv_and_res_block(64, 128, 2)
        self.cr_block3 = make_conv_and_res_block(128, 256, 8)
        self.cr_block4 = make_conv_and_res_block(256, 512, 8)
        self.cr_block5 = make_conv_and_res_block(512, 1024, 4)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.cr_block1(tmp)
        tmp = self.cr_block2(tmp)
        out3 = self.cr_block3(tmp)
        out2 = self.cr_block4(out3)
        out1 = self.cr_block5(out2)

        return out1, out2, out3


class YoloNetTail(nn.Module):

    """The tail side of the YoloNet.
    It will take the result from DarkNet53BackBone and do some upsampling and concatenation.
    It will finally output the detection result.
    Assembling YoloNetTail and DarkNet53BackBone will give you final result"""

    def __init__(self):
        super(YoloNetTail, self).__init__()
        self.detect1 = DetectionBlock(1024, 1024, 'l', 32)
        self.conv1 = ConvLayer(512, 256, 1)
        self.detect2 = DetectionBlock(768, 512, 'm', 16)
        self.conv2 = ConvLayer(256, 128, 1)
        self.detect3 = DetectionBlock(384, 256, 's', 8)

    def forward(self, x1, x2, x3):
        out1 = self.detect1(x1)
        branch1 = self.detect1.branch
        tmp = self.conv1(branch1)
        tmp = F.interpolate(tmp, scale_factor=2)
        tmp = torch.cat((tmp, x2), 1)
        out2 = self.detect2(tmp)
        branch2 = self.detect2.branch
        tmp = self.conv2(branch2)
        tmp = F.interpolate(tmp, scale_factor=2)
        tmp = torch.cat((tmp, x3), 1)
        out3 = self.detect3(tmp)

        return out1, out2, out3


class YoloNetV3(nn.Module):

    def __init__(self, nms=False, post=True):
        super(YoloNetV3, self).__init__()
        self.darknet = DarkNet53BackBone()
        self.yolo_tail = YoloNetTail()
        self.nms = nms
        self._post_process = post

    def forward(self, x):
        tmp1, tmp2, tmp3 = self.darknet(x)
        out1, out2, out3 = self.yolo_tail(tmp1, tmp2, tmp3)
        out = torch.cat((out1, out2, out3), 1)
        logging.debug("The dimension of the output before nms is {}".format(out.size()))
        return out

    def yolo_last_layers(self):
        _layers = [self.yolo_tail.detect1.conv7,
                   self.yolo_tail.detect2.conv7,
                   self.yolo_tail.detect3.conv7]
        return _layers

    def yolo_last_two_layers(self):
        _layers = self.yolo_last_layers() + \
                  [self.yolo_tail.detect1.conv6,
                   self.yolo_tail.detect2.conv6,
                   self.yolo_tail.detect3.conv6]
        return _layers

    def yolo_last_three_layers(self):
        _layers = self.yolo_last_two_layers() + \
                  [self.yolo_tail.detect1.conv5,
                   self.yolo_tail.detect2.conv5,
                   self.yolo_tail.detect3.conv5]
        return _layers

    def yolo_tail_layers(self):
        _layers = [self.yolo_tail]
        return _layers

    def yolo_last_n_layers(self, n):
        try:
            n = int(n)
        except ValueError:
            pass
        if n == 1:
            return self.yolo_last_layers()
        elif n == 2:
            return self.yolo_last_two_layers()
        elif n == 3:
            return self.yolo_last_three_layers()
        elif n == 'tail':
            return self.yolo_tail_layers()
        else:
            raise ValueError("n>3 not defined")
