#!/usr/bin/env bash

CONFIG=output/finetune_qqp/config.yaml
GPUS=1
NODE=1
NODE_RANK=0
PORT=2345

python3 -m oneflow.distributed.launch \
    --nproc_per_node $GPUS \
    --nnodes $NODE \
    --node_rank $NODE_RANK \
    --master_addr $PORT \
    projects/QQP/finetune_qqp.py \
    --config-file $CONFIG \
    --num-gpus $GPUS