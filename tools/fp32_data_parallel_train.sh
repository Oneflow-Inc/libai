# 8卡数据并行, 关闭amp
bash tools/train.sh tools/train_net.py configs/vit_imagenet.py 8 train.amp.enabled=False

# 4卡数据并行
bash tools/train.sh tools/train_net.py configs/vit_imagenet.py 4 train.amp.enabled=False

# 2卡数据并行
bash tools/train.sh tools/train_net.py configs/vit_imagenet.py 2 train.amp.enabled=False