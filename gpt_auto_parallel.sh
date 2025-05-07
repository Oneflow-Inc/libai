NUM_DEVICES=${1:-8}
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ONEFLOW_DEBUG=1
export ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE=0



# debug branch of auto parallel

# DDP
# bash tools/train.sh tools/train_net.py projects/libai-parallel-case/configs/gpt2_pretrain_data_parallel.py $NUM_DEVICES train.train_iter=200  train.log_period=1 graph.enabled=True optim.fused=True train.dist.device_type="npu" train.input_placement_device="npu" train.amp.enabled=False model.cfg.scale_mask_softmax_fusion=False model.cfg.embedding_dropout_prob=0.0 model.cfg.attention_dropout_prob=0.0 model.cfg.bias_gelu_fusion=False

# Auto Parallel
#export GLOG_v=3
#export GLOG_log_dir=./logs
#export GLOG_v=5
#export GLOG_logtostderr=1
export AUTO_MEMORY_MODE=HeavyMemoryDown
bash tools/train.sh tools/train_net.py \
    projects/libai-parallel-case/configs/gpt2_pretrain_auto_parallel.py $NUM_DEVICES \
    train.log_period=1 \
    graph.enabled=True \
    optim.fused=True \
    train.dist.device_type="npu" \
    train.input_placement_device="npu" \
    train.train_micro_batch_size=4 \
    train.train_iter=10 \
    train.amp.enabled=False \
    model.cfg.scale_mask_softmax_fusion=False \
    model.cfg.embedding_dropout_prob=0.0 \
    model.cfg.attention_dropout_prob=0.0 \
    model.cfg.bias_gelu_fusion=False
