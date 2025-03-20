DP=${1:-1}
TP=${2:-1}
PP=${3:-1}
NUM_DEVICES=$(( DP * TP * PP ))
#export ASCEND_RT_VISIBLE_DEVICES=1,4,5,6,7
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=2
#export ONEFLOW_DEBUG=1
export ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE=0
python3 -m oneflow.distributed.launch \
    --nproc_per_node $NUM_DEVICES \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 18245 \
        tools/train_net.py --config-file=configs/bert_large_pretrain.py \
            graph.enabled=True \
	    optim.fused=True \
            train.input_placement_device="npu" \
            train.dist.device_type="npu" \
            train.amp.enabled=False \
	    train.evaluation.enabled=False \
	    train.log_period=1 \
	    train.dist.data_parallel_size=$DP \
            train.dist.tensor_parallel_size=$TP \
            train.dist.pipeline_parallel_size=$PP \
	    model.cfg.hidden_dropout_prob=0.0 \
            model.cfg.attention_probs_dropout_prob=0.0 \
	    train.train_micro_batch_size=32 \
	    train.train_iter=10 \
	    model.cfg.bias_dropout_fusion=False \
	    model.cfg.scale_mask_softmax_fusion=False \
            model.cfg.bias_gelu_fusion=False
	    #train.train_iter=10 \
	    #train.load_weight=init_1b_ckpt \
        #tools/train_net.py --config-file=configs/gpt2_pretrain.py \
        #tools/train_net.py --config-file=projects/libai-parallel-case/configs/gpt2_pretrain_data_parallel.py \
	    #train.load_weight=init_1b_ckpt \
	    #train.num_accumulation_steps=2 \
	    #train.train_micro_batch_size=1 \
	    #train.num_accumulation_steps=2 \
	    #train.train_iter=10 \
            #optim.lr=0.0 \
