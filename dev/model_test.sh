#!/bin/bash -e

# cd to libai project root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export ONEFLOW_TEST_DEVICE_NUM=4

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m unittest -f tests/models/test_bert.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m unittest -f tests/models/test_gpt.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m unittest -f tests/models/test_t5.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m unittest -f tests/models/test_vit.py