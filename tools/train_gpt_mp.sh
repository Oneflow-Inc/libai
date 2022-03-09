#!/usr/bin/env bash

# CONFIG=$1
GPUS=$1
NODE=${NODE:-1}
NODE_RANK=${NODE_RANK:-0}
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-12345}

python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS --nnodes $NODE --node_rank $NODE_RANK --master_addr $ADDR --master_port $PORT \
tools/train_net.py \
--config-file configs/gpt2_pretrain.py \
    model.cfg.embedding_dropout_prob=0.1 \
    model.cfg.attention_dropout_prob=0.1 \
    model.cfg.num_attention_heads=16 \
    model.cfg.hidden_size=384 \
    model.cfg.ffn_hidden_size=1536 \
    model.cfg.num_layers=1 \
    model.cfg.max_seq_length=1024 