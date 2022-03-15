# 8卡数据并行
bash tools/train.sh tools/train_net.py configs/vit_imagenet.py 8

# 4卡数据并行
bash tools/train.sh tools/train_net.py configs/vit_imagenet.py 4

# 2卡数据并行
bash tools/train.sh tools/train_net.py configs/vit_imagenet.py 2