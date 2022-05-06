# SimCSE
Contributor{Xie Zipeng: xzpaiks@163.com}

Reproduce Supervised-simcse and Unsupervised-simcse with OneFlow.

## Introduce
SimCSE is a sentence representation learning method, in which there are two training methods: supervised learning and unsupervised learning. The unsupervised learning method is to input sentences and predict itself in the comparison target, and only use the standard dropout as noise; The supervised learning method uses the NLI data set, taking 'entry' as a positive sample and 'contrast' as a negative sample for supervised learning. This task uses Spearman to evaluate the model's performance on STS dataset, and uses Alignment and Uniformity to measure the effect of contrastive learning. 
- 《SimCSE: Simple Contrastive Learning of Sentence Embeddings》: https://arxiv.org/pdf/2104.08821.pdf
- Official GitHub: https://github.com/princeton-nlp/SimCSE

## Modle List(single GPU)
learning_rate=3e-5, batch_size=64
|      Unsupervised-Model        |STS-B dev |STS-B test|Pool type |
|:-------------------------------|:--------:|:--------:|:--------:|
|[unsup-simcse-bert-base-chinese](http://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/LiBai/SimCSE/unsup-simcse-bert-base-chinese.zip)  |74.64     |68.67     |cls           |
|unsup-simcse-bert-base-chinese  |74.86     |68.71     |last-avg      |
|unsup-simcse-bert-base-chinese  |64.33     |54.82     |pooled        |
|unsup-simcse-bert-base-chinese  |74.32     |67.55     |first-last-avg|

learning_rate=1e-5, batch_size=64
|       Supervised-Model         |STS-B dev |STS-B test|Pool type |
|:-------------------------------|:--------:|:--------:|:--------:|
|[sup-simcse-bert-base-chinese](http://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/LiBai/SimCSE/sup-simcse-bert-base-chinese.zip)    |80.93     |77.32     |cls         |
|sup-simcse-bert-base-chinese    |81.20     |77.09     |last-avg    |
|sup-simcse-bert-base-chinese    |76.61     |75.00     |pooled      |
|sup-simcse-bert-base-chinese    |80.64     |76.33     |first-last-avg|

## Training
Training SimCSE on 8 GPUs using data parallelism.
```bash
cd /path/to/libai
bash projects/SimCSE/train.sh tools/train_net.py projects/SimCSE/configs/config_simcse_unsup.py 8
```

## Evaluation
Evaluate SimCSE on 8 GPUs using data parallelism:
```bash
cd /path/to/libai
bash projects/SimCSE/train.sh tools/train_net.py projects/SimCSE/configs/config_simcse_unsup.py 8 --eval-only
```
