# Training & Evaluation in Command Line 

LiBai provides multiple arguments for covering a variety of situations.

- [Training command line](##Training)
- [Evaluation command line](##Evaluation)
- [Quickly check in the respective loop](##Quickly-check-in-the-respective-loop)

## Training

LiBai provides `tools/train.sh` and `tools/train_net.py` for launching training & eval task.

You can modify `tools/train_net.py` according to your own needs.

### Training & totally evaluation

For completely train and test, you can run: 

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                    # number of gpus
path_to_your_config.py # config.py for your task
```

### Training & partly evaluation 

If the test dataset costs a lot of time, you can set `train.evaluation.eval_iter=20` in your `config.py` or in the command line, it will run 20 steps for only part of the test dataset in testing for fast eval:

> NOTE: the eval metric will be calculated in partly testing dataset

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                             # number of gpus
path_to_your_config.py \        # config.py for your task
train.evaluation.eval_iter=20   # set eval_iter for testing
```

### Training & no evaluation

If you want to train without evaluation, you can set `train.evaluation.enabled=False` in your `config.py` or in the command line:

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                              # number of gpus
path_to_your_config.py \         # config.py for your task
train.evaluation.enabled=False   # set no evaluation 
```

### Resume train

If you want to resume training, you should set `--resume` in the command line, and set `train.output_dir` in your `config.py` or in the command line

For example: your training was interrupted unexpectly, your lastest model path is `output/demo/model_0000019/`. you should set `train.output_dir=output/demo` for resume trainig.

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                              # number of gpus
path_to_your_config.py \         # config.py for your task
--resume \                       # set resume training
train.output_dir=path/task       # set resume path, it should be parent directory of model path
```


## Evaluation

If you want to evaluate your model without training,  you should set `--eval-only` in your command line, and set `train.load_weight`.

Besides, `train.evaluation.eval_iter=20` is also valid in `--eval-only`, you can set it according to your own needs.

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                                          # number of gpus
path_to_your_config.py \                     # config.py for your task
--eval-only \                                # set eval without train
train.load_weight=path/task/model_final      # set model path
```

## Quickly check in the respective loop

If you want to quickly run several batches of train, eval and test to find any bugs, you can set `--fast-dev-run` in command line. It will change config settings to
```python
train.train_epoch = 0
train.train_iter = 20
train.evaluation.eval_period = 10
train.log_period = 1
```
runing command, `train.evaluation.eval_iter=20` is also valid in `--fast-dev-run `, you can set it according to own your needs.
```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                                          # number of gpus
path_to_your_config.py \                     # config.py for your task
--fast-dev-run                               # set for quickly check
``` 
