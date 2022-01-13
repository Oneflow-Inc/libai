#!/bin/bash

IMPL=cached

python tools/preprocess_data.py \
        --input data/test_sample.json \
        --vocab bert-vocab.txt \
        --dataset-impl ${IMPL} \
        --tokenizer-name BertTokenizer \
        --do-lower-case \
        --split-sentences \
        --output-prefix test_samples_${IMPL} \
        --workers 1 \
        --log-interval 2