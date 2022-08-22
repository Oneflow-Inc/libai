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
from collections import OrderedDict

from nltk.translate.bleu_score import corpus_bleu

from libai.utils import distributed as dist

from .evaluator import DatasetEvaluator


class BLEUEvaluator(DatasetEvaluator):
    """
    Evaluate BLEU(Bilingual Evaluation Understudy) score.

    BLEU is a score for comparing a candidate translation
    of text to one or more reference translations.
    """

    def __init__(self):
        super().__init__()
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        candidate = outputs["candidate"]
        reference = inputs["reference"]

        self._predictions.append({"candidate": candidate, "reference": reference})

    def evaluate(self):
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions

        candidates = []
        references = []
        for pred in predictions:
            candidates.append(pred["candidate"])
            references.append(pred["reference"])

        bleu_score = corpus_bleu(references, candidates)

        self._results = OrderedDict()
        self._results["bleu_score"] = bleu_score

        return copy.deepcopy(self._results)
