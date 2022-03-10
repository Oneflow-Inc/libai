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
import math
from collections import OrderedDict

from libai.utils import distributed as dist

from .evaluator import DatasetEvaluator


class PPLEvaluator(DatasetEvaluator):
    """
    Evaluate perplexity for Language Model.

    Perplexity is a measurement of how well a probability distribution or
    probability model predicts a sample.
    """

    def __init__(self):
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for k, v in outputs.items():
            ppl = math.exp(min(20, v.item()))
            self._predictions.append({f"{k}_PPL": ppl})

    def evaluate(self):
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions

        self._results = OrderedDict()
        for prediction in predictions:
            for k, v in prediction.items():
                if k not in self._results:
                    self._results[k] = 0
                self._results[k] += v

        for k in self._results.keys():
            self._results[k] /= len(predictions)

        return copy.deepcopy(self._results)
