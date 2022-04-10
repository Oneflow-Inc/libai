#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NODE=${NODE:-1}
NODE_RANK=${NODE_RANK:-0}
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-12345}
DATA_PATH="data"

if [ ! -d "$DATA_PATH" ]; then
    wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/LiBai/projects/simcse/data.zip
    unzip data.zip
    wget https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt -P ./data
    wget https://huggingface.co/bert-base-chinese/resolve/main/pytorch_model.bin -P ./data
fi

python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS --nnodes $NODE --node_rank $NODE_RANK --master_addr $ADDR --master_port $PORT \
./train_net.py \
--config-file $CONFIG ${@:3}
