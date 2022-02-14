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
from collections import OrderedDict, defaultdict

import oneflow as flow

from libai.utils import distributed as dist

from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


# def accuracy(output, target, topk=1):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     # output: [128]
#     # target: [128, 1000]
#     with flow.no_grad():
#         # TODO: support tuple topk=(1, 5, 10)
#         # maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(topk, 1, True, True)
#         pred = pred.transpose(0, 1)
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         # TODO: support tuple topk
#         # res = []
#         # for k in topk:
#         #     correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         #     res.append(correct_k.mul_(100.0 / batch_size))
#         correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
#         res = correct_k.mul_(100.0 / batch_size).item()
#         return res

def accuracy(output, target, topk=(1, )):
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [(correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size).item() for k in topk]


class ClsEvaluator(DatasetEvaluator):
    def __init__(self, cfg):
        self.cfg = cfg
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        pred_logits = outputs["prediction_scores"]
        labels = inputs["labels"]

        # measure accuracy
        topk_acc = accuracy(pred_logits, labels, topk=self.cfg.train.topk)
        num_correct_acc_topk = [acc * labels.size(0) / 100 for acc in topk_acc]

        self._predictions.append({"num_correct_topk": num_correct_acc_topk, "num_samples": labels.size(0)})

    def evaluate(self):
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions

        total_correct_num = OrderedDict()
        for top_k in self.cfg.train.topk:
            total_correct_num["Acc@"+str(top_k)] = 0
        
        total_samples = 0
        for prediction in predictions:
            for top_k, num_correct_n in zip(self.cfg.train.topk, prediction["num_correct_topk"]):
                total_correct_num["Acc@"+str(top_k)] += int(num_correct_n)

            total_samples += int(prediction["num_samples"])
        
        self._results = OrderedDict()
        for top_k, topk_correct_num in total_correct_num.items():
            self._results[top_k] = topk_correct_num / total_samples * 100

        return copy.deepcopy(self._results)
