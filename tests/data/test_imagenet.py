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

from libai.config import LazyConfig
from libai.data.imagenet import build_dataset

cfg = LazyConfig.load("./configs/common/data/imagenet_data.py")
# set path to imagenet
cfg.data.data_path = "/DATA/disk1/ImageNet/extract/"

# test train dataset
train_dataset, nb_classes = build_dataset(is_train=True, cfg=cfg)
assert len(train_dataset) == 1281167
assert nb_classes == 1000

# test valid/test dataset
valid_dataset, nb_classes = build_dataset(is_train=False, cfg=cfg)
assert len(valid_dataset) == 50000
assert nb_classes == 1000
