# Training & Evaluation in Command Line 

LiBai provides multiple arguments for covering a variety of situations.

- [training command line](##trainin)
- [evaluation command line](##evaluation)
- [quickly check total pipeline](##quickly-check-total-pipeline)

## training

LiBai provides `tools/train.sh` and `tools/train_net.py` for launching training & eval command.

You can modify `tools/train_net.py` according to your needs.

### train & totally evaluation

For completely train and test, you can run: 

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                    # number of gpus
path_to_your_config.py # config.py for your task
```

### train & partly evaluation 

If test dataset cost much time. you can set `train.evaluation.eval_iter=20` in your `config.py` or in command line, it will run `20` steps in testing for fast eval:

> NOTE: the eval metric will be calculated in partly testing dataset

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                             # number of gpus
path_to_your_config.py \        # config.py for your task
train.evaluation.eval_iter=20   # set eval_iter for testing
```

### train & no evaluation

If you want to train without evaluation, you can set `train.evaluation.enabled=False` in your `config.py` or in command line:

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                              # number of gpus
path_to_your_config.py \         # config.py for your task
train.evaluation.enabled=False   # set no evaluation 
```

### resume train

If you want to resume traing, you should set `--resume` in command line, and set `train.output_dir` in your `config.py` and command line

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                              # number of gpus
path_to_your_config.py \         # config.py for your task
--resume \                       # set resume training
train.output_dir=path/task       # set resume path, it should be parent directory of model path
```


## evaluation

If you want to evaluate your model without train,  you should set `--eval-only` in you command line, and set `train.load_weight`.

Besides, `train.evaluation.eval_iter=20` is also valid in `eval-only`, you can set it according to your needs.

```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                                          # number of gpus
path_to_your_config.py \                     # config.py for your task
--eval-only \                                # set eval without train
train.load_weight=path/task/model_final      # set model path
```

## quickly check total pipeline

If you want to quickly run several batches of train, eval and test to find any bugs, you can set `--fast-dev-run` in command line. It will change config settings to
```python
train.train_epoch = 0
train.train_iter = 20
train.evaluation.eval_period = 10
train.log_period = 1
```
runing command, `train.evaluation.eval_iter=20` is also valid in `--fast-dev-run `, you can set it according to your needs.
```shell
bash tools/train.sh \
tools/train_net.py \ 
4 \                                          # number of gpus
path_to_your_config.py \                     # config.py for your task
--fast-dev-run                               # set for quickly check
``` 
