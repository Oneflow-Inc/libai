DP=${1:-1}
TP=${2:-1}
PP=${3:-1}
NUM_DEVICES=$(( DP * TP * PP ))
python3 -m oneflow.distributed.launch \
    --nproc_per_node $NUM_DEVICES \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 18245 \
        tools/train_net.py --config-file=configs/gpt2_pretrain.py \
            graph.enabled=True \
	    train.dist.data_parallel_size=$DP \
            train.dist.tensor_parallel_size=$TP \
            train.dist.pipeline_parallel_size=$PP \
	    train.train_micro_batch_size=4 \
	    train.train_iter=10 \
	    train.log_period=1
