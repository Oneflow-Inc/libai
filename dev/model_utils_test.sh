#!/bin/bash -e

# cd to libai project root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TEST_OUTPUT=output_unittest
export ONEFLOW_TEST_DEVICE_NUM=4

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/model_utils/test_bert_utils.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/model_utils/test_roberta_utils.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/model_utils/test_gpt_utils.py

rm -rf $TEST_OUTPUT
