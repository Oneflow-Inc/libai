#!/usr/bin/env bash

CONFIG=configs/vit_imagenet.py #output/your_task/config.yaml
GPUS=4
NODE=1
NODE_RANK=0
ADDR=127.0.0.1

pkill python3
python3 -m oneflow.distributed.launch \
    --nproc_per_node $GPUS \
    --nnodes $NODE \
    --node_rank $NODE_RANK \
    --master_addr $ADDR \
    tools/train_net.py \
    --config-file $CONFIG \
    