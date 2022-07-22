
# 4 GPUs, fp16, 4 workers
python3 -m oneflow.distributed.launch \
--nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
tools/train_net.py \
--config-file projects/idea_t5/configs/config.py \
dataloader.train.num_workers=4 \
train.amp.enabled=False \
train.output_dir="test_3_5/4g_4p_fp32_megatrondata"
