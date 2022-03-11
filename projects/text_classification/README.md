# Text classification projects
**English** | [简体中文](/projects/text_classification/README_zh-CN.md)
## GLUE tasks

Fine-tuning the pretrained language models for sequence classification on the GLUE benchmark: [General Language Understanding Evaluation](https://gluebenchmark.com/). This is a common benchmark for *English* pretrained language models.

GLUE is made up of a total of 9 different tasks. The tasks contains cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli. You can perform the following command to download and extract the dataset:
```bash
cd projects/text_classification
python3 dataset/download_glue_data.py
```

Here is how to finetune the task on one of them:
```bash
bash tools/train.sh tools/train_net.py projects/text_classification/configs/config.py $num_gpus train.train_iter=10
```
where `$num_gpus` indicates the number of GPUs. If you want to run the distributed program, you can change it, for example:
```bash
bash tools/train.sh tools/train_net.py projects/text_classification/configs/config.py 2 train.train_iter=10
```

Before running the program, you should modify the `config.py` file. Modification fields include but are not limited to task_name, data_dir, vocab_file, model hyperparameter, learning rate, and batch size, and so on.

## CLUE tasks

For *Chinese* pretrained language model, it can be evaluated by [CLUE benchmark](https://github.com/CLUEbenchmark/CLUE). This is a sequence classification task just like GLUE, so you can finetune your model with the same command. Don't forget to modify `config.py` file.

You can download and extract the datasets by:
```bash
cd projects/text_classification
python3 dataset/download_clue_data.py
```