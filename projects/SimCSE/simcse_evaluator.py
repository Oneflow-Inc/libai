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

import copy
import logging
from collections import OrderedDict
import numpy as np
from libai.utils import distributed as dist

from .evaluator import DatasetEvaluator
from scipy.stats import spearmanr


def spearman_target(labels, cos_sim):
    return spearmanr(labels, cos_sim.cpu().numpy()).correlation


class SimcseEvaluator(DatasetEvaluator):
    def __init__(self, cfg):
        self.cfg = cfg
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        cos_sim = outputs["cos_sim"]             # [batch]
        labels = inputs["labels"]                # [batch]  

        # measure spearman
        res = spearman_target(labels, cos_sim)

        self._predictions.append(
            {"cos_sim": cos_sim, "labels":labels}
        )

    def evaluate(self):
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions

        sim_tensor = flow.tensor([])
        label_array = np.array([])

        for prediction in predictions:
            sim_tensor = flow.cat((sim_tensor, prediction["spearman"]), dim=0)
            label_array = np.append(label_array, np.array(prediction["labels"].cpu().numpy()))
        self._results = spearman_target(label_array, sim_tensor)
        return copy.deepcopy(self._results, sim_tensor.cpu().numpy())
