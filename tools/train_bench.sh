
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# unset CUDA_VISIBLE_DEVICES

# export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1

export ONEFLOW_THREAD_ENABLE_LOCAL_MESSAGE_QUEUE=1
export ONEFLOW_KERNEL_DISABLE_BLOB_ACCESS_CHECKER=1
export ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR=1
export ONEFLOW_STREAM_REUSE_CUDA_EVENT=1


# 1 GPU, fp32
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# train.amp.enabled=False \
# train.output_dir="output/oneflow-0.7/1g_fp32"

# 1 GPU, fp16
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# train.amp.enabled=True \
# train.output_dir="output/oneflow-0.7/1g_fp16"

# 8 GPUs, fp32, data parallel
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# train.amp.enabled=False \
# train.output_dir="output/oneflow-0.7/8g_fp32_data-parallel"

# 8 GPUs, fp16, data parallel
python3 -m oneflow.distributed.launch \
--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
tools/train_net.py \
--config-file configs/t5_test.py \
train.train_iter=20 \
train.amp.enabled=False \
train.output_dir="output/oneflow-0.7/8g_fp16_data-parallel"

# 8 GPUs, fp32, tensor parallel
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# train.amp.enabled=False \
# train.dist.tensor_parallel_size=8 \
# train.output_dir="output/oneflow-0.7/8g_fp32-tensor-parallel"

# 8 GPUs, fp16, tensor parallel
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# train.amp.enabled=True \
# train.dist.tensor_parallel_size=8 \
# train.output_dir="output/oneflow-0.7/8g_fp16-tensor-parallel"

# 8 GPUs, fp32, hybrid parallel
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# train.amp.enabled=False \
# train.dist.tensor_parallel_size=2 \
# train.output_dir="output/oneflow-0.7/8g_fp32-hybrid-parallel"

# 8 GPUs, fp16, hybrid parallel
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# train.amp.enabled=True \
# train.dist.tensor_parallel_size=2 \
# train.output_dir="output/oneflow-0.7/8g_fp16-hybrid-parallel"

# 8 GPUs, fp32, 3D parallel
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# train.amp.enabled=False \
# train.dist.tensor_parallel_size=2 \
# train.dist.pipeline_parallel_size=2 \
# train.output_dir="output/oneflow-0.7/8g_fp32_3d-parallel"

# 8 GPUs, fp32, 3D parallel
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# train.amp.enabled=True \
# train.dist.tensor_parallel_size=2 \
# train.dist.pipeline_parallel_size=2 \
# train.output_dir="output/oneflow-0.7/8g_fp16_3d-parallel"

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

# # 4 GPUs, fp32, 8 workers, hybrid parallel
# python3 -m oneflow.distributed.launch \
# --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=8 \
# train.amp.enabled=False \
# train.dist.tensor_parallel_size=2 \
# train.output_dir="output/4g_8p_fp32_hybrid"

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
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_test.py \
# dataloader.train.num_workers=4 \
# train.amp.enabled=True \
# train.dist.tensor_parallel_size=1 \
# train.output_dir="output/new_of_test_8g_fp16_env_amp-embed"
