# Training & Evaluation in Command Line 

LiBai provides multiple arguments for covering a variety of situations.

## Training

LiBai provides `tools/train.sh` and `tools/train_net.py` for launching training & eval task.

You can modify `tools/train_net.py` according to your own needs.

### Training & Evaluation

For completely train and test, you can run: 

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                    # number of gpus
path_to_your_config.py # config.py for your task
```

### Training & Partial Evaluation 

If the evaluation process is time consuming, you can set the parameter `train.evaluation.eval_iter` in your `config.py` to a smaller number such as 20, which can make the evaluation process faster by only using part of the testset. You can also set the parameter by the command line directly :

> NOTE: the eval metric will be calculated by part of testing dataset

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                             # number of gpus
path_to_your_config.py \        # config.py for your task
train.evaluation.eval_iter=20   # set eval_iter for testing
```

### Training & No Evaluation

If you want to train without evaluation, you can set `train.evaluation.enabled=False` in your `config.py` or in the command line:

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                              # number of gpus
path_to_your_config.py \         # config.py for your task
train.evaluation.enabled=False   # set no evaluation 
```

### Resume Training

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

Besides, `train.evaluation.eval_iter=20` will be valid in `--eval-only` if you set it, you can set `eval_iter` according to your own needs.

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                                          # number of gpus
path_to_your_config.py \                     # config.py for your task
--eval-only \                                # set eval without train
train.load_weight=path/task/model_final      # set model path
```

## Quickly check in the respective loop

If you want to find out whether there are any bugs in your program, you can pass `--fast-dev-run` to the command line, which will change config settings to:
```python
train.train_epoch = 0
train.train_iter = 20
train.evaluation.eval_period = 10
train.log_period = 1
```
Besides, `train.evaluation.eval_iter=20` will be valid in `--fast-dev-run` if you set it, you can set `eval_iter` according to your own needs.
```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                                          # number of gpus
path_to_your_config.py \                     # config.py for your task
--fast-dev-run                               # set for quickly check
``` 
