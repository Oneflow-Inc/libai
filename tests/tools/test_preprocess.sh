#!/bin/bash

IMPL=lazy

python tools/preprocess_data.py \
        --input data/test_sample_cn.json \
        --vocab-file bert-base-chinese-vocab.txt \
        --dataset-impl ${IMPL} \
        --tokenizer-name BertTokenizer \
        --do-lower-case \
        --do-chinese-wwm \
        --split-sentences \
        --output-prefix cn_samples_${IMPL} \
        --workers 1 \
        --log-interval 2