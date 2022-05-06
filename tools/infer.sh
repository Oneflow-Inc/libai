#!/usr/bin/env bash

FILE=$1
GPUS=$2
NODE=${NODE:-1}
NODE_RANK=${NODE_RANK:-0}
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-12345}

python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS --nnodes $NODE --node_rank $NODE_RANK --master_addr $ADDR --master_port $PORT \
$FILE ${@:3}

