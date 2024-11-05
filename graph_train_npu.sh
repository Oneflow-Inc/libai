export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=0
#export ONEFLOW_DEBUG=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
        tools/train_net.py --config-file=configs/gpt2_pretrain.py \
            graph.enabled=True \
            train.input_placement_device="npu" \
            train.dist.device_type="npu" \
            train.amp.enabled=False \
	    train.evaluation.enabled=False \
	    train.log_period=1 \
	    train.load_weight=init_1b_ckpt \
            model.cfg.scale_mask_softmax_fusion=False \
	    model.cfg.embedding_dropout_prob=0.0 \
            model.cfg.attention_dropout_prob=0.0 \
            model.cfg.bias_gelu_fusion=False
	    #train.train_iter=10 \
	    #train.train_micro_batch_size=1 \
	    #train.train_micro_batch_size=2 \
	    #train.num_accumulation_steps=2 \
	    #train.train_iter=10 \
            #optim.lr=0.0 \
