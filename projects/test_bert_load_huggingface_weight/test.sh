#!/usr/bin/env bash

PRETRAINED_PATH="./bert-base-chinese"

if [ ! -d "$PRETRAINED_PATH" ]; then
    wget https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt -P ./bert-base-chinese/
    wget https://huggingface.co/bert-base-chinese/resolve/main/pytorch_model.bin -P ./bert-base-chinese/
    wget https://huggingface.co/bert-base-chinese/resolve/main/config.json -P ./bert-base-chinese/
fi

python3 test_output.py