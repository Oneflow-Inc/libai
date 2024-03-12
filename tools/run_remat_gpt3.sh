#!/usr/bin/env bash

set -ux

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

MODEL_NAME=gpt3
RESULT_DIR=./exp_data

array=(32500 39000 45500 52000 58500 65000)
config_file=/share_nfs/sd_dataset/lph/codes/libai_normal/configs/dtr_gpt3_pretrain.py

METHOD=ours
for threshold in "${array[@]}"; do
    CUDA_VISIBLE_DEVICES=0 ONEFLOW_REMAT_SUMMARY_FILE_PREFIX=${RESULT_DIR}/$MODEL_NAME-$METHOD-$threshold ENABLE_PROFILE_FOR_DTR=0 ONEFLOW_VM_MULTI_THREAD=0 ONEFLOW_DTR_GROUP_NUM=2 \
    python -m oneflow.distributed.launch \
    --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
    tools/train_remat.py --config-file $config_file --threshold $threshold --fast-dev-run
done

METHOD=dte-our-impl
for threshold in "${array[@]}"; do
    CUDA_VISIBLE_DEVICES=0 ONEFLOW_REMAT_SUMMARY_FILE_PREFIX=${RESULT_DIR}/$MODEL_NAME-$METHOD-$threshold ENABLE_PROFILE_FOR_DTR=0 ONEFLOW_VM_MULTI_THREAD=0 ONEFLOW_DTR_GROUP_NUM=1 ONEFLOW_REMAT_HEURISTIC_DTE=1 \
    python -m oneflow.distributed.launch \
    --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
    tools/train_remat.py --config-file $config_file --threshold $threshold --fast-dev-run
done

METHOD=dtr-no-free
for threshold in "${array[@]}"; do
    CUDA_VISIBLE_DEVICES=0 ONEFLOW_REMAT_SUMMARY_FILE_PREFIX=${RESULT_DIR}/$MODEL_NAME-$METHOD-$threshold ENABLE_PROFILE_FOR_DTR=0 ONEFLOW_VM_MULTI_THREAD=0 ONEFLOW_DTR_GROUP_NUM=1 ONEFLOW_REMAT_HEURISTIC_DTR=1 \
    python -m oneflow.distributed.launch \
    --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
    tools/train_remat.py --config-file $config_file --threshold $threshold --fast-dev-run
done