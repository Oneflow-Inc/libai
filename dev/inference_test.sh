#!/bin/bash -e

# cd to libai project root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TEST_OUTPUT=output_unittest
export ONEFLOW_TEST_DEVICE_NUM=4

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/inference/test_text_generation.py 

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/inference/test_text_classification.py 

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/inference/test_image_classification.py

rm -rf $TEST_OUTPUT
