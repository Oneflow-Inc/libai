#! /bin/bash

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export ONEFLOW_DEBUG_MODE=True

# DATASET=/dataset/Megatron-LM/dummy/gpt_sample_dataset_text_document
DATASET=/dataset/gpt/gpt_sample_dataset_text_document
SEQ_LEN=1024
LAYER_NUM=12
HIDDEN_SIZE=768
HEAD_NUM=12
MBZ=8
GBZ=8
ACC_STEPS=1
TMP=1
PMP=1
TRAIN_ITER=300
LOG_INTERVAL=1
SAVE_PATH=gpt2_model
SRC_DIR=$(realpath $(dirname "$0")/..)

# gdb --args \
python3 $SRC_DIR/train.py \
    --data-type gpt_dataset \
    --model-type gpt-2 \
    --dataset $DATASET \
    --split 949,50,1 \
    --vocab-size 50257 \
    --max-seq-length $SEQ_LEN \
    --num-layers $LAYER_NUM \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $HEAD_NUM \
    --micro-batch-size $MBZ \
    --global-batch-size $GBZ \
    --tensor-model-parallel-size $TMP \
    --pipeline-model-parallel-size $PMP \
    --checkpoint-activations \
    --num-gpus-per-node 1 \
    --num-nodes 1 \
    --optimizer adamw \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --initial-loss-scale 4294967296 \
    --learning-rate 0.001 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters 320000 \
    --lr-warmup-fraction 0.01 \
    --fp16 \
    --graph \
    --train-iters $TRAIN_ITER \
    --log-interval $LOG_INTERVAL \
    --save-path $SAVE_PATH
