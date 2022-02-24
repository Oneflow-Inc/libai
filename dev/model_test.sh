#!/bin/bash -e

# cd to libai project root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export ONEFLOW_TEST_DEVICE_NUM=4
python3 -m oneflow.distributed.launch --nproc_per_node 4 -m unittest discover -f -v -s ./tests/models
