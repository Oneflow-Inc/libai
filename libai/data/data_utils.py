# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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


import time
import logging
import numpy as np
import oneflow as flow
from .indexed_dataset import make_dataset as make_indexed_dataset

logger = logging.getLogger(__name__)

def get_indexed_dataset(data_prefix, data_impl, skip_warmup):

    logger.info("building dataset index ...")

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
    assert indexed_dataset.sizes.shape[0] == indexed_dataset.doc_idx[-1]
    logger.info(
        "inished creating indexed dataset in {:4f} "
        "seconds".format(time.time() - start_time)
    )

    logger.info("indexed dataset stats:")
    logger.info(
        "number of documents: {}".format(indexed_dataset.doc_idx.shape[0] - 1)
    )
    logger.info("number of sentences: {}".format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


def get_prefixes_and_weights(data_prefix):
    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [0] * num_datasets
    prefixes = [0] * num_datasets
    weight_sum = 0.0
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2 * i])
        weight_sum += weights[i]
        prefixes[i] = (data_prefix[2 * i + 1]).strip()
    
    # Normalize weights
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]

    return prefixes, weights


def compile_helper():
    """Compile helper function at runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys
        sys.exit(1)

