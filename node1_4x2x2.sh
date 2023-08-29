addr=${1:-192.168.40.21}
NODE=2 NODE_RANK=1 ADDR=${addr} PORT=12345 bash tools/train.sh \
	tools/train_net.py \
	configs/bert_large_pretrain_4x2x2.py 8
