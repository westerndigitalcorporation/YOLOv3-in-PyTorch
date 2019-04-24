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


import json

from PIL import Image
import torch
from torch.utils.data import Dataset

from config import NUM_ATTRIB
from .transforms import default_transform_fn, random_transform_fn
from utils import xywh_to_cxcywh


class CaltechPedDataset(Dataset):

    training_set = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']
    validation_set = ['set06', 'set07', 'set08', 'set09', 'set10']

    def __init__(self, root, img_size, video_set='all', transform='default'):
        # set video sets used in this dataset
        if video_set == 'all':
            self.video_set_str = self.training_set + self.validation_set
        elif video_set == 'training':
            self.video_set_str = self.training_set
        elif video_set == 'validation':
            self.video_set_str = self.validation_set
        elif isinstance(video_set, list):
            all_set = self.training_set + self.validation_set
            if isinstance(video_set[0], int):
                video_set = self.video_set_int2str(video_set)
            for set_name in video_set:
                assert (set_name in all_set)
            self.video_set_str = video_set
        else:
            raise TypeError("video_set should be one of the following: "
                            "'all', 'training', 'validation' or a list of custom sets.")
        self.video_set_int = self.video_set_str2int(self.video_set_str)  # get int version of video set
        # get the number of the images in each of the set
        n_imgs_all_set = json.load(open("{}/attributes.json".format(root)))
        self.n_imgs_per_video_loaded_set = [n_imgs_all_set[i] for i in self.video_set_int]
        self.n_imgs_aggregated_loaded_set = [sum(l) for l in self.n_imgs_per_video_loaded_set]
        self.root = root
        self.annotations = json.load(open("{}/annotations.json".format(root)))

        if transform == 'default':
            self._tf = default_transform_fn(img_size)
        elif transform == 'random':
            self._tf = random_transform_fn(img_size)
        else:
            raise ValueError("input transform can only be 'default' or 'random'.")

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("the dataset query is out of range.")
        img_i = index
        set_i = -1
        for set_i, n_imgs in enumerate(self.n_imgs_aggregated_loaded_set):
            if img_i < n_imgs:
                break
            img_i -= n_imgs
        n_imgs_per_video = self.n_imgs_per_video_loaded_set[set_i]
        video_i = -1
        for video_i, n_imgs in enumerate(n_imgs_per_video):
            if img_i < n_imgs:
                break
            img_i -= n_imgs
        set_i_str = "set{:02}".format(set_i)
        video_i_str = "V{:03}".format(video_i)
        img_i_str = str(img_i)
        img_file_path = "{}/data/images/{}/{}/{:05}.png".format(self.root, set_i_str, video_i_str, img_i)
        try:
            targets = self.annotations[set_i_str][video_i_str]['frames'][img_i_str]
        except KeyError:
            targets = []
        img = Image.open(img_file_path)
        # tensor_transform = tv_tf.ToTensor()
        # img_tensor = tensor_transform(img)
        labels = []
        for target in targets:
            bbox = torch.tensor(target['pos'], dtype=torch.float32) # in xywh format
            # category_id = target['category_id']
            # if (not self.all_categories) and (category_id != self.category_id):
            #     continue
            # one_hot_label = coco_category_to_one_hot(category_id, dtype='float32')
            # conf = torch.tensor([1.])
            label = torch.cat((bbox, torch.tensor([1., 1.])))
            labels.append(label)
        if labels:
            label_tensor = torch.stack(labels)
        else:
            label_tensor = torch.zeros((0, NUM_ATTRIB))
        # if targets:
        #     label_position_tensor = torch.Tensor([elem['pos'] for elem in targets])
        # else:
        #     label_position_tensor = torch.zeros((0, NUM_ATTRIB))
        img, label_tensor = self._tf(img, label_tensor)
        label_tensor = xywh_to_cxcywh(label_tensor)
        return img, label_tensor, label_tensor.size(0)

    def __len__(self):
        return sum(self.n_imgs_aggregated_loaded_set)

    @staticmethod
    def video_set_str2int(video_set_str):
        return list(map(lambda x: int(x[3:]), video_set_str))

    @staticmethod
    def video_set_int2str(video_set_int):
        return list(map(lambda x: "set{:02}".format(x), video_set_int))