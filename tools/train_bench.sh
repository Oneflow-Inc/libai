
export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 -m oneflow.distributed.launch \
--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
tools/train_net.py \
--config-file configs/t5_test.py train.output_dir="output/t5_dp"

# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_profile.py train.dist.tensor_parallel_size=8 \
# train.output_dir="output/t5_mp"

# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_profile.py train.dist.tensor_parallel_size=2 \
# train.output_dir="output/t5_mix"

# python3 -m oneflow.distributed.launch \
# --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
# tools/train_net.py \
# --config-file configs/t5_profile.py train.dist.tensor_parallel_size=2 train.dist.pipeline_parallel_size=2 \
# train.output_dir="output/t5_3d"
