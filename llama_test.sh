#!/bin/bash

DP=${1:-1}
TP=${2:-1}
PP=${3:-1}
NUM_DEVICES=$(( DP * TP * PP ))

BATCH_SIZE=${4:-8}

# 模型规模（如 7b, 13b）
MODEL_SIZE=${5:-7b}

# 是否启用自动并行（1 = 是，0 = 否）
AUTO_PARALLEL=${6:-1}
export PRETRAINED_MODEL_PATH=meta-llama/Llama-2-${MODEL_SIZE}-hf

if [ "$AUTO_PARALLEL" -eq 1 ]; then
    CONFIG_FILE=projects/Llama/configs/llama_sft_ap.py
else
    CONFIG_FILE=projects/Llama/configs/llama_sft.py
fi

#export ONEFLOW_NPU_COMM_SYNC=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node $NUM_DEVICES \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 18245 \
    projects/Llama/train_net.py \
        --config-file=${CONFIG_FILE} \
        graph.enabled=True \
        train.input_placement_device="npu" \
        train.dist.device_type="npu" \
        train.amp.enabled=False \
	train.num_accumulation_steps=1 \
        train.dist.data_parallel_size=$DP \
        train.dist.tensor_parallel_size=$TP \
        train.dist.pipeline_parallel_size=$PP \
        train.train_micro_batch_size=$BATCH_SIZE \
	train.train_epoch=0 \
	train.train_iter=10 \
	load_weights=False \
	train.log_period=1
    #tools/train_net.py \
