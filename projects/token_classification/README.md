Here is how to finetune the task on one of them:
```bash
bash tools/train.sh tools/train_net.py projects/token_classification/configs/config.py 1 train.train_iter=10
```


命名实体识别数据集
1、CLUENER2020：https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/cluener_public 
2、MSRA：https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/MSRA 
3、人民网（04年）：https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/people_daily 
4、微博命名实体识别数据集：https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/weibo 
5、BosonNLP NER数据：https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/boson（2000条） 
6、影视-音乐-书籍实体标注数据：https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/video_music_book_datasets 
7、中文医学文本命名实体识别 2020CCKS：https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/2020_ccks_ner 
8、电子简历实体识别数据集:https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/ResumeNER 
9 、医渡云实体识别数据集:https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/yidu-s4k 
10、 简历实体数据集：https://github.com/jiesutd/LatticeLSTM/tree/master/data 
11、CoNLL-2003：https://www.clips.uantwerpen.be/conll2003/ner/ 
12、Few-NERD 细粒度数据集:https://github.com/thunlp/Few-NERD/tree/main/data ......