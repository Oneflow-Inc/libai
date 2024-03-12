#!/usr/bin/env bash

set -ux

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

MODEL_NAME=gpt2
RESULT_DIR=./exp_data_${MODEL_NAME}
THRESHOLD=9500

config_file=/home/dev/files/repos/libai-normal/libai/config/configs/${MODEL_NAME}_pretrain.py


CUDA_VISIBLE_DEVICES=0 ONEFLOW_REMAT_SUMMARY_FILE_PREFIX=${RESULT_DIR}/$MODEL_NAME-checkpointing ENABLE_PROFILE_FOR_DTR=0 ONEFLOW_VM_MULTI_THREAD=0 ONEFLOW_DTR_GROUP_NUM=2 \
python -m oneflow.distributed.launch \
--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
tools/train_remat.py --config-file $config_file --threshold $THRESHOLD --fast-dev-run