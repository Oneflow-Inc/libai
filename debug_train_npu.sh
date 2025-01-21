# export ASCEND_RT_VISIBLE_DEVICES=0,1
# train.train_iter=20


# eager 1n1d
python3 tools/train_net_npu.py --config-file configs/gpt2_pretrain_npu.py graph.enabled=False
# graph 1n1d
# python3 tools/train_net_npu.py --config-file configs/gpt2_pretrain_npu.py graph.enabled=True

# eager ddp 1nXg
# nohup  python3 -m oneflow.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345   tools/train_net.py --config-file=configs/gpt2_pretrain_npu.py  graph.enabled=False   train.input_placement_device="npu"   train.dist.device_type="npu"  train.amp.enabled=False   train.evaluation.enabled=False  train.log_period=20  model.cfg.scale_mask_softmax_fusion=False    model.cfg.embedding_dropout_prob=0.0   model.cfg.attention_dropout_prob=0.0  model.cfg.bias_gelu_fusion=False > debug_eager_train_npu_without_dropout.log 2>&1  &

# graph ddp 1nXg
# nohup python3 -m oneflow.distributed.launch  --nproc_per_node 2  --nnodes 1  --node_rank 0  --master_addr 127.0.0.1  --master_port 12345  tools/train_net.py --config-file=configs/gpt2_pretrain_npu.py  graph.enabled=True  train.input_placement_device="npu"  train.dist.device_type="npu"  train.amp.enabled=False  train.evaluation.enabled=False   train.log_period=20  model.cfg.scale_mask_softmax_fusion=False  model.cfg.embedding_dropout_prob=0.0 model.cfg.attention_dropout_prob=0.0  model.cfg.bias_gelu_fusion=False  train.train_iter=40 > debug_graph_train_npu.log 2>&1 &

# grapg ddp 1nXg 2D parallel
# nohup ./graph_train_npu.sh 2 2 > 2d_debug.log 2>&1 &
# nohup python3 -m oneflow.distributed.launch  --nproc_per_node 4  --nnodes 1  --node_rank 0  --master_addr 127.0.0.1  --master_port 12345  tools/train_net.py --config-file=configs/gpt2_pretrain_npu_2d_parallel.py  graph.enabled=True  train.input_placement_device="npu"  train.dist.device_type="npu"  train.amp.enabled=False  train.evaluation.enabled=False   train.log_period=20  model.cfg.scale_mask_softmax_fusion=False  model.cfg.embedding_dropout_prob=0.0 model.cfg.attention_dropout_prob=0.0  model.cfg.bias_gelu_fusion=False  train.train_iter=40 > debug_graph_train_npu_2d_parallel_241210.log 2>&1 &
