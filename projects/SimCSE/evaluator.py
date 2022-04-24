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

import numpy as np
from scipy.stats import spearmanr

from libai.evaluation import DatasetEvaluator
from libai.utils import distributed as dist


def spearman_target(cos_sim, labels):
    return spearmanr(cos_sim, labels).correlation


class SimcseEvaluator(DatasetEvaluator):
    def __init__(self):
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        sim = outputs["sim"]
        labels = inputs["labels"]
        self._predictions.append({"sim": sim, "labels": labels})

    def evaluate(self):
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions
        sim_array = np.array([])
        label_array = np.array([])
        for prediction in predictions:
            sim_array = np.append(sim_array, dist.tton(prediction["sim"]))
            label_array = np.append(label_array, dist.tton(prediction["labels"]))
        self._results = spearman_target(sim_array, label_array)
        return {"Spearman": self._results}
