#!/bin/bash -e

# cd to libai project root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TEST_OUTPUT=output_unittest
export ONEFLOW_TEST_DEVICE_NUM=4
export ONEFLOW_EP_CUDA_ENABLE_TF32_EXECUTION=0

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/model_loader/test_bert_loader.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/model_loader/test_roberta_loader.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/model_loader/test_gpt_loader.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/model_loader/test_mt5_loader.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/model_loader/test_t5_loader.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/model_loader/test_swin_loader.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/model_loader/test_swinv2_loader.py

python3 -m oneflow.distributed.launch --nproc_per_node 4 -m pytest -s --disable-warnings tests/model_loader/test_vit_loader.py

rm -rf $TEST_OUTPUT