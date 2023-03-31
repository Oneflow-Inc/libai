# MT5

Reproduce T5Model and MT5Model with OneFlow, which effect are equivalent to HuggingFace's [T5](https://huggingface.co/docs/transformers/v4.19.4/en/model_doc/t5#overview) and [T5v1.1](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511).

## Introduce
The t5 and mt5 pretraining project can support 3D parallel and [ZERO](https://arxiv.org/abs/2202.10435).

## Training MT5
Training MT5 on 8 GPUs using 3D parallelism and ZERO.

### 1. Prepare your training config file

> set the pretrain parameters in `MT5/configs/mt5_pretrain.py`, such as `vocab_file` and `data_prefix`.

> If you would like to use the t5 model, please set `model_type`="t5".

### 2. Prepare the demo training data

Prepare the demo training data by running:
```bash
# path/to/libai
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/bert-base-chinese-vocab.txt -P ./data_test/bert_data/
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.bin -P ./data_test/bert_data/
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.idx -P ./data_test/bert_data/
```

### 3. Prepare your own training data

If you want to use your own training data, please skip the step2, and refer [Preprocessing Dataset](https://libai.readthedocs.io/en/latest/tutorials/basics/Preprocessing_Dataset.html#).

```bash
IMPL=mmap
KEYS=text

python tools/preprocess_data.py \
        --input /path/to/libai/projects/MT5/data/test.json \
        --json-keys ${KEYS} \
        --vocab-file /path/to/libai/projects/MT5/data/vocab.txt \
        --dataset-impl ${IMPL} \
        --tokenizer-name BertTokenizer \
        --do-lower-case \
        --do-chinese-wwm \
        --split-sentences \
        --output-prefix magic_prompt_${IMPL} \
        --workers 4 \
        --log-interval 2

```

### 4. Run the following code to start training
```bash
# cd /path/to/libai
bash tools/train.sh projects/MT5/train_net.py projects/MT5/configs/mt5_pretrain.py 8
```
