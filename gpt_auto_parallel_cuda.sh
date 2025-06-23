NUM_DEVICES=${1:-8}
bash tools/train.sh tools/train_net.py \
    projects/libai-parallel-case/configs/gpt2_pretrain_auto_parallel.py $NUM_DEVICES \
    train.train_micro_batch_size=4 \
    train.train_iter=10 \
    train.log_period=1
