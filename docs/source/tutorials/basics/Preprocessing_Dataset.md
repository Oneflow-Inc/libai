# Preprocessing Dataset

If you use LiBai's `Dataset` to training NLP model, you can preprocess the training data.

This tutorial introduces how to preprocess your own training data, let's take training `Bert` as an example.

First, You need to store the training data in loose JSON format file, which contains one text sample per line, For example:

```bash
{"chapter": "Chapter One", "text": "April Johnson had been crammed inside an apartment", "type": "April", "background": "novel"}
{"chapter": "Chapter Two", "text": "He couldn't remember their names", "type": "Dominic", "background": "novel"}
```

You can set the `--json-keys` argument to select the specific data of per sample, and the other keys will not be used.

Then, Process the JSON file into a binary format for training. To conver the json into mmap, cached index file, or the lazy loader format use `toos/preprocess_data.py`. Set the `--dataset-impl` flag to `mmap`, `cached`, or `lazy` respectively. You can run the following code to prepare you own dataset for training BERT:

```bash
#!/bin/bash

IMPL=mmap
KEYS=text

python tools/preprocess_data.py \
        --input path/to/test_sample_cn.json \
        --json-keys ${KEYS} \
        --vocab-file path/to/bert-base-chinese-vocab.txt \
        --dataset-impl ${IMPL} \
        --tokenizer-name BertTokenizer \
        --do-lower-case \
        --do-chinese-wwm \
        --split-sentences \
        --output-prefix cn_samples_${IMPL} \
        --workers 1 \
        --log-interval 2
```

Further command line arguments are described in the source file [`preprocess_data.py`](https://github.com/Oneflow-Inc/libai/blob/main/tools/preprocess_data.py).
