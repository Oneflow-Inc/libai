#!/usr/bin/env bash

CONFIG=configs/vit_base_patch16_224.py #output/your_task/config.yaml
GPUS=1
NODE=1
NODE_RANK=0
PORT=2345

python3 -m oneflow.distributed.launch \
    --nproc_per_node $GPUS \
    --nnodes $NODE \
    --node_rank $NODE_RANK \
    --master_addr $PORT \
    tools/train_vit.py \
    --config-file $CONFIG \