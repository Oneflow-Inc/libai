LEARNING_RATE=0.001
MOM=0.9
EPOCH=10
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=64


python3 cifar10_example.py \
    --learning_rate $LEARNING_RATE \
    --mom $MOM \
    --epochs $EPOCH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \