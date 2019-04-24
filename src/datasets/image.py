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


import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transforms import default_transform_fn


def _get_padding(h, w):
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


class ImageFolder(Dataset):
    """The ImageFolder Dataset class."""

    def __init__(self, folder_path, img_size=416, sort_key=None):
        self.files = sorted(glob.glob('{}/*.*'.format(folder_path)), key=sort_key)
        self.img_shape = (img_size, img_size)
        self._img_size = img_size
        self._transform = default_transform_fn(img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        max_size = max(w, h)
        _padding = _get_padding(h, w)
        # Add padding
        # _transform = default_transform_fn(_padding, self._img_size)
        transformed_img_tensor, _ = self._transform(img)

        scale = self._img_size / max_size

        return img_path, transformed_img_tensor, scale, np.array(_padding)

    def __len__(self):
        return len(self.files)
