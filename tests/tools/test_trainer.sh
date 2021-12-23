#! /bin/bash

source $1

_DEVICE_NUM_PER_NODE=2
_MASTER_ADDR=127.0.0.1
_NUM_NODES=1
_NODE_RANK=0

run_cmd="python3 -m oneflow.distributed.launch \
    --nproc_per_node $_DEVICE_NUM_PER_NODE \
    --nnodes $_NUM_NODES \
    --node_rank $_NODE_RANK \
    --master_addr $_MASTER_ADDR \
    tests/test_trainer.py"

echo ${run_cmd}
eval ${run_cmd}

set +x