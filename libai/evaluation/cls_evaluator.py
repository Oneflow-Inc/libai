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

import oneflow as flow

from libai.utils import distributed as dist

from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


def accuracy(output, target, topk=1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with flow.no_grad():
        # TODO: support tuple topk=(1, 5, 10)
        # maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(topk, 1, True, True)
        pred = pred.transpose(0, 1)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # TODO: support tuple topk
        # res = []
        # for k in topk:
        #     correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
        res = correct_k.mul_(100.0 / batch_size).item()
        return res


class ClsEvaluator(DatasetEvaluator):
    def __init__(self, cfg):
        self.cfg = cfg
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        # FIX ME: support dict args, not implement in graph right now
        pred_logits = outputs[-1]  # decide by your model output
        labels = inputs[-1]  # decide by your dataloder output

        # measure accuracy
        acc1 = accuracy(pred_logits, labels, topk=1)
        num_correct_acc1 = acc1 * labels.size(0) / 100

        self._predictions.append({"num_correct": num_correct_acc1, "num_samples": labels.size(0)})

    def evaluate(self):
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions

        total_correct_num = 0
        total_samples = 0
        for prediction in predictions:
            total_correct_num += prediction["num_correct"]
            total_samples += prediction["num_samples"]

        acc1 = total_correct_num / total_samples * 100

        self._results = OrderedDict()
        self._results["Acc@1"] = acc1

        return copy.deepcopy(self._results)
