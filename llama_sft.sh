#size=${1:-7b}
#model_path=/data/hf_models/Llama-2-13b-hf
#model_path=/data/hf_models/Llama-2-${size}-hf

#export ONEFLOW_NPU_COMM_SYNC=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 18245 \
    projects/Llama/train_net.py \
        --config-file=projects/Llama/configs/llama_sft.py \
        graph.enabled=True \
        train.input_placement_device="npu" \
        train.dist.device_type="npu" \
        train.amp.enabled=False \
	train.num_accumulation_steps=1 \
	train.train_epoch=0 \
	train.train_iter=20 \
	train.log_period=1
    #tools/train_net.py \
