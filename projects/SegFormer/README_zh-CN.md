# segformer in libai
segformer是transformer在语义分割任务上的经典应用，是一种简单、有效且鲁棒性强的语义分割的方法。SegFormer 由两部分组成：(1) 层次化Transformer Encoder (2) 仅由几个FC构成的decoder；

segformer仅在ImageNet1K上进行预训练，再在下游任务语义分割数据集上进行finetune训练,如 ADE20K, Cityscapes, Coco Stuff.
## 项目构成
1. 相关类
- SegformerPreTrainModel
- SegformerModel
- SegformerForImageClassification
- SegformerForSemanticSegmentation
- SegformerSegmentationLoadImageNetPretrain
- DecodeHead
- SegFLoaderHuggerFace
- SegFLoaderLiBai
- SegFLoaderImageNet1kPretrain

2. 各类间的关系及作用
- SegformerPreTrainModel是整个项目的base class 也是预训练的backbone
- SegformerModel继承SegformerPreTrainModel，是分割任务的backbone，两者的不同点知识在forward过程中对输入的处理方式不同，以便适应不同的任务
- SegformerForImageClassification 是用于segformer用于分类的class，由SegformerPreTrainModel作为backbone以及一个简单的linear layer作为head，若用户想自己进行预训练，则可用该类进行预训练
- SegformerForSemanticSegmentation 是用于分割的class，由SegformerModel作为backbone以及DecodeHead作为head，若用户想自己进行预训练，则可用该类进行预训练
- SegformerSegmentationLoadImageNetPretrain 也是用于分割的类，若用户想加载官方提供的预训练权重，则只需设置cfg.pretrained_model_path为想加载权重的文件夹
- DecodeHead是用于分割的head
- SegFLoaderHuggerFace：工具类，用于加载huggingface Segformer权重至 LiBai SegformerForSemanticSegmentation
- SegFLoaderLiBai:工具类：加载LiBai相关权重
- SegFLoaderImageNet1kPretrain:工具类，用于加载官方提供的ImageNet1K预训练权重

3. 文件结构
```
SegFormer
│   README.md
│   README_zh-CN.md    
│
└───configs //各种配置相关
│   │   segformer_city.py //加载pretrain在cityscape上训练的配置文件
│   │   segformer_imgnet1k_pretrain.py //imagenet预训练的配置文件
│   │
│   └───data
│   │       citiscapes.py //cityscape数据集配置文件
│   └───models  //模型配置文件
│       mit_b0-b5.py //不同规模的配置文件
│       classification //分类(预训练)不同规模的配置文件
│
└───dataset //数据集及其增强相关文件    
│
└───model_utils //权重加载相关文件
│       segf_loader.py //加载huggingface或libai的模型权重
│       load_pretrained_imagenet1k.py //加载official imagenet1k pertained weight
│
└───modeling //模型文件
│       head.py // 分割头模型
│       segformer_model.py // 模型主文件，包含大部分相关类
│       segformer_loadmodel.py // 加载官方预训练权重的类，用于加载权重进行finetune
│
└───pretrained  //存放预训练权重
```

## 训练
1. 加载official imagenet1k pretrain weight 在分割任务数据集如cityscapes上进行训练
下载在Imagenet1K上预训练的权重 [google drive](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing)|[one drive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ),放在pretrained文件夹, 并修改权重文件名为pytorch_model.bin
```
bash tool/train.sh tool/train_net.py projects/Segformer/configs/segformer_city.py 4
```
默认为4卡的数据并行， 可自定义修改
支持：
- 数据并行/模型并行/流水并行/数据+模型2D并行/其他并行可自行测试
改变dataloader 即可在不同的分割数据集上使用


2. 加载huggingface/libai checkpoint 进行推理验证或进行精读对齐
```
python projects/Segformer/model_utils/test_segloader.py
```


3. 进行ImageNet预训练
```
bash tool/train.sh tool/train_net.py projects/Segformer/configs/segformer_imgnet1k.py 4
```
默认为4卡的数据并行， 可自定义修改
支持：
- 数据并行/模型并行/流水并行/数据+模型2D并行/其他并行可自行测试