#!/usr/bin/env bash

set -ux

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

MODEL_NAME=gpt2
RESULT_DIR=./exp_data1
# array=(4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500)
array=(3400 4250 5100 5950 6800 7650 8500)
# array=(3500 4000 4500 5000 5500 6000 6500 7500 8500 9500)
config_file=/home/dev/files/repos/libai-normal/libai/config/configs/gpt2_pretrain.py

METHOD=ours
for threshold in "${array[@]}"; do
    CUDA_VISIBLE_DEVICES=2 ONEFLOW_REMAT_SUMMARY_FILE_PREFIX=${RESULT_DIR}/$MODEL_NAME-$METHOD-$threshold ENABLE_PROFILE_FOR_DTR=0 ONEFLOW_VM_MULTI_THREAD=0 ONEFLOW_DTR_GROUP_NUM=2 \
    python -m oneflow.distributed.launch \
    --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
    tools/train_remat.py --config-file $config_file --threshold $threshold --fast-dev-run
done

METHOD=dte-our-impl
for threshold in "${array[@]}"; do
    CUDA_VISIBLE_DEVICES=2 ONEFLOW_REMAT_SUMMARY_FILE_PREFIX=${RESULT_DIR}/$MODEL_NAME-$METHOD-$threshold ENABLE_PROFILE_FOR_DTR=0 ONEFLOW_VM_MULTI_THREAD=0 ONEFLOW_DTR_GROUP_NUM=1 ONEFLOW_REMAT_HEURISTIC_DTE=1 \
    python -m oneflow.distributed.launch \
    --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
    tools/train_remat.py --config-file $config_file --threshold $threshold --fast-dev-run
done

METHOD=dtr-no-free
for threshold in "${array[@]}"; do
    CUDA_VISIBLE_DEVICES=2 ONEFLOW_REMAT_SUMMARY_FILE_PREFIX=${RESULT_DIR}/$MODEL_NAME-$METHOD-$threshold ENABLE_PROFILE_FOR_DTR=0 ONEFLOW_VM_MULTI_THREAD=0 ONEFLOW_DTR_GROUP_NUM=1 ONEFLOW_REMAT_HEURISTIC_DTR=1 \
    python -m oneflow.distributed.launch \
    --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
    tools/train_remat.py --config-file $config_file --threshold $threshold --fast-dev-run
done