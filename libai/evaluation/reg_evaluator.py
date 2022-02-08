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
import oneflow as flow
from scipy.stats import pearsonr, spearmanr

from libai.utils import distributed as dist

from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class RegEvaluator(DatasetEvaluator):
    def __init__(self, cfg):
        self.cfg = cfg
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        # FIX ME: support dict args, not implement in graph right now
        preds = outputs[-1]  # decide by your model output
        labels = inputs[-1]  # decide by your dataloder output

        # measure accuracy
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        self._predictions.append({"preds": preds, "labels": labels})

    def evaluate(self):
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions

        preds = np.array([])
        labels = np.array([])
        for prediction in predictions:
            preds = np.concatenate(preds, prediction["preds"])
            labels = np.concatenate(labels, prediction["labels"])

        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        corr = (pearson_corr + spearman_corr) / 2

        self._results = OrderedDict()
        self._results["pearson"] = pearson_corr
        self._results["spearman"] = spearman_corr
        self._results["corr"] = corr

        return copy.deepcopy(self._results)
