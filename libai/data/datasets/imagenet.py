# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import oneflow as flow
from oneflow.utils.data import DataLoader

from flowvision import datasets, transforms
from flowvision.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flowvision.data import Mixup
from flowvision.data import create_transform
from flowvision.transforms.functional import str_to_interp_mode

from libai.data.structures import DistTensorData, Instance


# def build_imagenet_dataset(is_train, cfg):
#     transform = build_transform(is_train, cfg)
#     prefix = "train" if is_train else "val"
#     root = os.path.join(cfg.data.data_path, prefix)
#     dataset = datasets.ImageFolder(root, transform=transform)
#     if is_train:
#         assert len(dataset) == 1281167, "The whole train set of ImageNet contains 1281167 images but got {} instead.".format(len(dataset))
#     else:
#         assert len(dataset) == 50000, "The whole val set of ImageNet contains 50000 images but got {} instead.".format(len(dataset))
#     nb_classes = 1000    
#     return dataset, nb_classes

class ImageNetDataset(datasets.ImageFolder):
    def __init__(self, is_train, cfg):
        # set train mode and cfg
        self.is_train = is_train
        self.cfg = cfg
        # set data-path
        prefix = "train" if is_train else "val"
        self.root = os.path.join(cfg.data.data_path, prefix)
        # set transform
        self.transform = build_transform(is_train, cfg)
    
    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        data_sample = Instance(
            images = DistTensorData(sample, placement_idx=0),
            targets = DistTensorData(target, placement_idx=-1)
        )
        return data_sample


def build_transform(is_train, cfg):
    resize_im = cfg.data.img_size > 32
    if is_train:
        transform = create_transform(
            input_size=cfg.data.img_size,
            is_training=True,
            color_jitter=cfg.data.augmentation.color_jitter if cfg.data.augmentation.color_jitter > 0 else None,
            auto_augment=cfg.data.augmentation.auto_augment if cfg.data.augmentation.auto_augment != 'none' else None,
            re_prob=cfg.data.augmentation.reprob,
            re_mode=cfg.data.augmentation.remode,
            re_count=cfg.data.augmentation.recount,
            interpolation=cfg.data.interpolation,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(cfg.data.img_size, padding=4)
    
    t = []
    if resize_im:
        if cfg.data.test.crop:
            size = int((256 / 224) * cfg.data.img_size)
            t.append(
                transforms.Resize(size, interpolation=str_to_interp_mode(cfg.data.interpolation))
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(cfg.data.img_size))
        else:
            t.append(
                transforms.Resize((cfg.data.img_size, cfg.data.img_size),
                                   interpolation=str_to_interp_mode(cfg.data.interpolation))
            )
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)