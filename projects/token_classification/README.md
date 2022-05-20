# Example of push project to LiBai
## Quick RUN Here is how to finetune the task on one of them:
```bash
bash tools/train.sh tools/train_net.py projects/token_classification/configs/config.py 1 train.train_iter=10
```

## 3 steps:
### step 1-config.py
- 使用get_config()方法 快速获得基本配置
```python
tokenization = get_config("common/data/bert_dataset.py").tokenization
optim = get_config("common/optim.py").optim
model_cfg = get_config("common/models/bert.py").cfg
graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train
```
- 为tokenizer配合vocal
```python
tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="/workspace/CQL_BERT/libai/projects/QQP/QQP_DATA/bert-base-chinese-vocab.txt",
    do_lower_case=True,
    do_chinese_wwm=False,
)
```
### dataset的构建，使其满足模型的输入输出
- 首先在config.py下引入特定dataset
```python
from projects.token_classification.dataset import CnerDataset
```
- 接着读取改dataset，转化为dataloader对象
```python
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(CnerDataset)(
            task_name="cner",
            data_dir="/workspace/CQL_BERT/libai/projects/token_classification/data/cner/cner",
            tokenizer=tokenization.tokenizer,
            max_seq_length=512,
            mode="train",
        ),
    ],
    num_workers=4,
)
```
- 接着就是编辑dataset
dataset 的指定传入格式如下，可根据不同模型微调。于是需要根据特定的数据集设定不同的 feature 形成格式。
```python
    def __getitem__(self, i):
        feature = self.features[i]
        return Instance(
            input_ids = DistTensorData(flow.tensor(feature.input_ids, dtype=flow.long)),
            attention_mask = DistTensorData(flow.tensor(feature.attention_mask, dtype=flow.long)),
            token_type_ids = DistTensorData(flow.tensor(feature.token_type_ids, dtype=flow.long)),
            labels = DistTensorData(flow.tensor(feature.labels, dtype=flow.long)),
        )
```
- feature 形成格式的specific的代码一般在 xxx_utils.py 下由 processor 产生的 example 作为参数，调用 `cner_convert_examples_to_features` 函数实现，此时如何根据数据集生成满足要求的 example 成为 `主要挑战`
以 `token_classification` 为例：
```python
    def get_labels(self):
        """See base class."""
        return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
                'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
                'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a,text_b=None, label=labels))
        return examples
```


