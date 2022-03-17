# QQP projects

QQP is a fine-tuning task for sequence classification, it belongs to GLUE: [General Language Understanding Evaluation](https://gluebenchmark.com/).

We provide code for finetuning on qqp dataset based on pretrained bert model. Differently, the language model is fine-tuned and pretrained from **Chinese** vocabulary. 

## Download Data

Run command:
```bash
python3 projects/QQP/dataset/download_qqp_data.py
```
Data and Vocabulary will be downloaded in `projects/QQP/QQP_DATA/`
> NOTE: pretrained model will not be provided due to privacy reasons
```
projects/QQP/QQP_DATA/
│---bert-base-chinese-vocab.txt
│---train.tsv    
│---dev.tsv
```

## Start Training
Run command:
```shell
bash tools/train.sh tools/train_net.py projects/QQP/configs/config_qqp.py $num_gpus
```
where `$num_gpus` indicates the number of GPUs. If you want to run the distributed program, you can change it, for example:
```bash
bash tools/train.sh tools/train_net.py projects/text_classification/configs/config.py 4
```

Before running the program, you should modify the `projects/QQP/configs/config_qqp.py` file. Modification fields include but are not limited to task_name, data_dir, vocab_file, model hyperparameter, learning rate, and batch size, and so on.