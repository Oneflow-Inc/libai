
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# unset CUDA_VISIBLE_DEVICES

# 1 GPU, fp32, 4 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=4 \
# train.amp.enabled=False \
# train.output_dir="output/1g_4p_fp32"

# # 1 GPU, fp32, 8 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=8 \
# train.amp.enabled=False \
# train.output_dir="output/1g_8p_fp32"

# # 1 GPU, fp16, 4 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=4 \
# train.amp.enabled=True \
# train.output_dir="output/1g_4p_fp16"

# # 1 GPU, fp16, 4 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=8 \
# train.amp.enabled=True \
# train.output_dir="output/1g_8p_fp16"

# # 4 GPUs, fp32, 4 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=4 \
# train.amp.enabled=False \
# train.output_dir="output/4g_4p_fp32"

# # 4 GPUs, fp32, 8 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=8 \
# train.amp.enabled=False \
# train.output_dir="output/4g_8p_fp32"

# # 4 GPUs, fp16, 4 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=4 \
# train.amp.enabled=True \
# train.output_dir="output/4g_4p_fp16"

# # 4 GPUs, fp16, 8 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=8 \
# train.amp.enabled=True \
# train.output_dir="output/4g_8p_fp16"

# # 8 GPUs, fp32, 4 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=4 \
# train.amp.enabled=False \
# train.output_dir="output/8g_4p_fp32"

# # 8 GPUs, fp32, 8 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=8 \
# train.amp.enabled=False \
# train.output_dir="output/8g_8p_fp32"

# # 8 GPUs, fp16, 4 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=4 \
# train.amp.enabled=True \
# train.output_dir="output/8g_4p_fp16"

# # 8 GPUs, fp16, 8 workers
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=8 \
# train.amp.enabled=True \
# train.output_dir="output/8g_8p_fp16"

# # ==============================

# # 4 GPUs, fp32, 8 workers, tensor parallel
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=8 \
# train.amp.enabled=False \
# train.dist.tensor_parallel_size=4 \
# train.output_dir="output/4g_8p_fp32_tensor"

# # 4 GPUs, fp16, 8 workers, tensor parallel
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=8 \
# train.amp.enabled=True \
# train.dist.tensor_parallel_size=4 \
# train.output_dir="output/4g_8p_fp16_tensor"

# 4 GPUs, fp32, 8 workers, hybrid parallel
python3 -m oneflow.distributed.launch \
--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
tools/train_net.py \
--config-file configs/t5_test.py \
dataloader.train.num_workers=4 \
train.amp.enabled=False \
train.train_iter=20 \
train.dist.tensor_parallel_size=1 \
train.output_dir="output/8g_4p_fp32"

# # 4 GPUs, fp16, 8 workers, hybrid parallel
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=8 \
# train.amp.enabled=True \
# train.dist.tensor_parallel_size=2 \
# train.output_dir="output/4g_8p_fp16_hybrid"

# python3 -m oneflow.distributed.launch \
# --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=2 \
# train.amp.enabled=True \
# train.dist.tensor_parallel_size=2 \
# train.output_dir="output/4g_2p_fp16_hybrid"
