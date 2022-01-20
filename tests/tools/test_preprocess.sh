#!/bin/bash

IMPL=lazy

python tools/preprocess_data.py \
        --input data/test_sample.json \
        --vocab-file spiece.model \
        --dataset-impl ${IMPL} \
        --tokenizer-name T5Tokenizer \
        --do-lower-case \
        --split-sentences \
        --output-prefix t5_samples_${IMPL} \
        --workers 1 \
        --log-interval 2