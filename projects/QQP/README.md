# QQP projects

QQP is a fine-tuning task for sequence classification, it belongs to GLUE: [General Language Understanding Evaluation](https://gluebenchmark.com/).

We provide code for finetuning on qqp dataset based on pretrained [Bert](https://arxiv.org/pdf/1810.04805.pdf) model. Differently, the language model is fine-tuned and pretrained from **Chinese** vocabulary. 

## Download Data

Run command:
```bash
cd /path/to/libai_root
python3 projects/QQP/dataset/download_qqp_data.py
```
Data and Vocabulary will be downloaded in `projects/QQP/QQP_DATA/`
> NOTE: pretrained model will not be provided due to privacy reasons, you can download pretrained bert model from [Megatron](https://github.com/NVIDIA/Megatron-LM)
```
projects/QQP/QQP_DATA/
│---bert-base-chinese-vocab.txt
│---train.tsv    
│---dev.tsv
```

## Start Training
Run command:
```shell
cd /path/to/libai_root
bash tools/train.sh tools/train_net.py projects/QQP/configs/config_qqp.py $num_gpus
```
where `$num_gpus` indicates the number of GPUs. For example, if you want to run the distributed program on 4 GPUs, you can set to 4.:
```bash
bash tools/train.sh tools/train_net.py projects/text_classification/configs/config.py 4
```

Before running the program, you should modify the `projects/QQP/configs/config_qqp.py` depend on your own needs. Modification fields include but are not limited to task_name, data_dir, vocab_file, model hyperparameter, learning rate, and batch size, and so on.