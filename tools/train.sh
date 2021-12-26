#!/usr/bin/env bash

CONFIG=$1
GPUS=${2:-1}
NODE=${NODE:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}

export CUDA_VISIBLE_DEVICES=3
# export PYTHONPATH=/workspace/oneflow/python:$PYTHONPATH

python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS --nnodes $NODE --node_rank $NODE_RANK --master_addr 127.0.0.1 \
tools/pretrain_bert.py \
--config-file $CONFIG --num-gpus $GPUS ${@:3}
