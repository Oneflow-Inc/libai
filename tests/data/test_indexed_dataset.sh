#!/bin/bash

python tests/data/test_indexed_dataset.py \
        --data test_samples_cached_text_sentence \
        --vocab bert-vocab.txt \
        --dataset-impl cached \
        --tokenizer-type BertWordPieceLowerCase


python tests/data/test_indexed_dataset.py \
        --data test_samples_lazy_text_sentence \
        --vocab bert-vocab.txt \
        --dataset-impl lazy \
        --tokenizer-type BertWordPieceLowerCase


python tests/data/test_indexed_dataset.py \
        --data test_samples_mmap_text_sentence \
        --vocab bert-vocab.txt \
        --dataset-impl mmap \
        --tokenizer-type BertWordPieceLowerCase

# cached, lazy: doc_idx: int64, sizes: int64, 不可使用 get, 对于这种，只能使用 SentenceIndexedDataset
# mmap: doc_idx: int64, sizes: int32, 可使用 get，对于这种，可以使用 DocumentIndexedDataset，也可以使用上一种


