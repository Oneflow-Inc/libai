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

import logging

from .data import GLUEAbstractDataset
from .data_utils import clean_text

logger = logging.getLogger("libai." + __name__)

LABELS = [0, 1]


class QQPDataset(GLUEAbstractDataset):
    def __init__(self, dataset_name, data_paths, tokenizer, max_seq_length, test_label=0):
        self.test_label = test_label
        self.dataset_name = dataset_name
        super().__init__("QQP", dataset_name, data_paths, tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """ "Implement abstract method."""
        logger.info(" > Processing {} ...".format(filename))

        samples = []
        total = 0
        first = True
        is_test = False
        with open(filename, "r") as f:
            for line in f:
                row = line.strip().split("\t")
                if first:
                    first = False
                    if len(row) == 3:
                        is_test = True
                        logger.info(
                            "   reading {}, {}, and {} columns and "
                            "setting labels to {}".format(
                                row[0].strip(), row[1].strip(), row[2].strip(), self.test_label
                            )
                        )
                    else:
                        assert len(row) == 6
                        logger.info(
                            "    reading {}, {}, {}, and {} columns"
                            " ...".format(
                                row[0].strip(), row[3].strip(), row[4].strip(), row[5].strip()
                            )
                        )
                    continue

                if is_test:
                    assert len(row) == 3, "expected length 3: {}".format(row)
                    uid = int(row[0].strip())
                    text_a = clean_text(row[1].strip())
                    text_b = clean_text(row[2].strip())
                    label = self.test_label
                    assert len(text_a) > 0
                    assert len(text_b) > 0
                else:
                    if len(row) == 6:
                        uid = int(row[0].strip())
                        text_a = clean_text(row[3].strip())
                        text_b = clean_text(row[4].strip())
                        label = int(row[5].strip())
                    else:
                        logger.info("***WARNING*** index error, " "skipping: {}".format(row))
                        continue
                    if len(text_a) == 0:
                        logger.info("***WARNING*** zero length a, " "skipping: {}".format(row))
                        continue
                    if len(text_b) == 0:
                        logger.info("***WARNING*** zero length b, " "skipping: {}".format(row))
                        continue
                assert label in LABELS
                assert uid >= 0

                sample = {"uid": uid, "text_a": text_a, "text_b": text_b, "label": label}
                total += 1
                samples.append(sample)

                if total % 50000 == 0:
                    logger.info("  > processed {} so far ...".format(total))

        logger.info(" >> processed {} samples.".format(len(samples)))
        return samples
