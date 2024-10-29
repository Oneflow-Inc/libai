export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
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
            model.cfg.scale_mask_softmax_fusion=False \
	    model.cfg.embedding_dropout_prob=0.0 \
            model.cfg.attention_dropout_prob=0.0 \
	    train.train_iter=10 \
	    train.log_period=1 \
            model.cfg.bias_gelu_fusion=False
