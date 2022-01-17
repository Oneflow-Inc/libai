#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NODE=${NODE:-1}
NODE_RANK=${NODE_RANK:-0}
ADDR=${ADDR:-127.0.0.1}

python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS --nnodes $NODE --node_rank $NODE_RANK --master_addr $ADDR \
tools/train_net.py \
--config-file $CONFIG ${@:3}
