#!/bin/bash

python tests/data/test_dataset.py \
        --data_prefix test_samples_mmap_text_sentence \
        --dataset-impl mmap \
        --vocab-file bert-vocab.txt \
        --tokenizer-type BertWordPieceLowerCase \
        --max-seq-length 30 \
        --mask-lm-prob 0.15;
        # --binary-head 
        # --append-eod 
