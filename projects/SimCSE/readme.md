# SimCSE
Reproduce Supervised-simcse and Unsupervised-simcse with OneFlow.

## Introduce
SimCSE is a sentence representation learning method, in which there are two training methods: supervised learning and unsupervised learning. The unsupervised learning method is to input sentences and predict itself in the comparison target, and only use the standard dropout as noise; The supervised learning method uses the NLI data set, taking 'entry' as a positive sample and 'contrast' as a negative sample for supervised learning. This task uses Spearman to evaluate the model's performance on STS dataset, and uses Alignment and Uniformity to measure the effect of contrastive learning. 
- 《SimCSE: Simple Contrastive Learning of Sentence Embeddings》: https://arxiv.org/pdf/2104.08821.pdf
- Official GitHub: https://github.com/princeton-nlp/SimCSE

## Evaluation(single GPU)
Dataset: SNLI+STS
|      Unsupervised-Model        |STS-B dev |STS-B test|Pool type |
|:-------------------------------|:--------:|:--------:|:--------:|
|unsup-simcse-bert-base-chinese  |74.64     |68.15     |cls           |
|unsup-simcse-bert-base-chinese  |74.86     |68.71     |last-avg      |
|unsup-simcse-bert-base-chinese  |64.33     |54.82     |pooled        |
|unsup-simcse-bert-base-chinese  |74.32     |67.55     |first-last-avg|

Dataset: SNLI
|       Supervised-Model         |STS-B dev |STS-B test|Pool type |
|:-------------------------------|:--------:|:--------:|:--------:|
|unsup-simcse-bert-base-chinese  |80.93     |77.24     |cls         |
|unsup-simcse-bert-base-chinese  |81.20     |77.09     |last-avg    |
|unsup-simcse-bert-base-chinese  |76.61     |75.00     |pooled      |
|unsup-simcse-bert-base-chinese  |80.64     |76.33     |first-last-avg|