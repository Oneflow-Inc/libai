Here is how to finetune the task on one of them:
```bash
bash tools/train.sh tools/train_net.py projects/token_classification/configs/config.py 1 train.train_iter=10
```