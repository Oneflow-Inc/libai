#!/bin/bash

IMPL=lazy
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