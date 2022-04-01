# Training & Evaluation in Command Line 

LiBai provides multiple arguments covering a variety of situations.

## Training

LiBai provides `tools/train.sh` and `tools/train_net.py` for launching training & eval task.

You can modify `tools/train_net.py` according to your own needs.

### Training & Evaluation

To completely train and test, run: 

```shell
bash tools/train.sh \
tools/train_net.py \ 
path_to_your_config.py \ # config.py for your task
4                        # number of gpus
```

### Training & Partial Evaluation 

If the evaluation process is time consuming, you can set the parameter `train.evaluation.eval_iter` in your `config.py` to a smaller number such as 20, which can make the evaluation process faster by using only part of the testset. You can also set the parameter by the command line directly :

**Note:** the eval metric will be calculated by part of testing dataset.

```shell
bash tools/train.sh \
tools/train_net.py \ 
path_to_your_config.py \        # config.py for your task
4 \                             # number of gpus
train.evaluation.eval_iter=20   # set eval_iter for testing
```

### Training & No Evaluation

To train without evaluation, set `train.evaluation.enabled=False` in your `config.py` or in the command line:

```shell
bash tools/train.sh \
tools/train_net.py \ 
path_to_your_config.py \         # config.py for your task
4 \                              # number of gpus
train.evaluation.enabled=False   # set no evaluation 
```

### Resume Training

To resume training, set `--resume` in the command line, and set `train.output_dir` in your `config.py` or in the command line

For example, if your training is interrupted unexpectly, and your lastest model path is `output/demo/model_0000019/`, then set `train.output_dir=output/demo` to resume trainig:

```shell
bash tools/train.sh \
tools/train_net.py \ 
path_to_your_config.py \         # config.py for your task
4 \                              # number of gpus
--resume \                       # set resume training
train.output_dir=path/task       # set resume path, it should be parent directory of model path
```


## Evaluation

To evaluate your model without training, set `--eval-only` in your command line, and set `train.load_weight`.

Besides, `train.evaluation.eval_iter=20` will be valid in `--eval-only` if you set it. You can set `eval_iter` according to your own needs.

```shell
bash tools/train.sh \
tools/train_net.py \ 
path_to_your_config.py \                     # config.py for your task
4 \                                          # number of gpus
--eval-only \                                # set eval without train
train.load_weight=path/task/model_final      # set model path
```

## Quickly check in the respective loop

To find out whether there are any bugs in your program, pass `--fast-dev-run` to the command line, which will change config settings to:
```python
train.train_epoch = 0
train.train_iter = 20
train.evaluation.eval_period = 10
train.log_period = 1
```
Besides, `train.evaluation.eval_iter=20` will be valid in `--fast-dev-run` if you set it. You can set `eval_iter` according to your own needs.
```shell
bash tools/train.sh \
tools/train_net.py \ 
path_to_your_config.py \                     # config.py for your task
4 \                                          # number of gpus
--fast-dev-run                               # set for quickly check
``` 
