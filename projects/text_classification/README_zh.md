
# 文本分类任务

## GLUE 任务

在文本分类任务上微调预训练语言模型。[GLUE](https://gluebenchmark.com/) 是一个英文文本分类常见的基准任务，通常用它评估预训练语言模型的理解能力。

GLUE 由9个子任务组成，包括cola、sst2、mrpc、stsb、qqp、mnli、qnli、rte、wnli。你可以执行通过命令实现一键下载并提取数据：
```bash
cd projects/text_classification
python3 dataset/download_glue_data.py
```

这里示范如何微调该任务：
```bash
bash tools/train.sh projects/text_classification/configs/config.py $num_gpus
```
其中`$num_gpus`表示运行程序的GPU数量。如果你想执行分布式训练，可以这样修改：
```bash
bash tools/train.sh projects/text_classification/configs/config.py 2
```

在运行程序之前，你应当修改`config.py`配置文件。待修改字段包括但不限于task_name、data_dir、vocab_file、模型超参数、学习率、批次大小等等。

## CLUE 任务

对于中文预训练语言模型，适用 CLUE 基准任务进行评估。和 GLUE 基准任务相同，CLUE 是一个文本分类任务，训练方法与上面相同。注意，记得修改`config.py`配置文件。

你可以下载并提取 CLUE 完整数据集通过以下命令:
```bash
python3 dataset/download_clue_data.py
```
