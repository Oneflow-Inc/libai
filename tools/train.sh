#!/usr/bin/env bash

FILE=$1
CONFIG=$2
GPUS=$3
NODE=${NODE:-1}
NODE_RANK=${NODE_RANK:-0}
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-12345}

export MULTIHEAD_ATTN_FUSION=true
# NOTE(chengcheng): temp disable fuse wait for zzk fix correctness bug.
#export ONEFLOW_FUSE_OPTIMIZER_UPDATE_CAST=true
unset ONEFLOW_FUSE_OPTIMIZER_UPDATE_CAST
export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=true


python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS --nnodes $NODE --node_rank $NODE_RANK --master_addr $ADDR --master_port $PORT \
$FILE --config-file $CONFIG ${@:4}
