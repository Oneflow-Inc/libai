# GLM

2017 年, Google 提出了 Transformer 架构, 随后 BERT 、GPT、T5等预训练模型不断涌现, 并在各项任务中都不断刷新 SOTA 纪录。去年, 清华提出了 GLM 模型(https://github.com/THUDM/GLM), 不同于上述预训练模型架构，它采用了一种自回归的空白填充方法, 在 NLP 领域三种主要的任务（自然语言理解、无条件生成、有条件生成）上都取得了不错的结果。

在LiBai中主要实现了GLM推理部分的工作，训练相关内容可以参考：

- [GLM国产大模型训练加速：性能最高提升3倍，显存节省1/3，低成本上手](https://mp.weixin.qq.com/s/dkTGXuJV38KuLb4_LmM20Q)
- https://github.com/Oneflow-Inc/one-glm


## GLM-Inference
当模型规模过于庞大，单个 GPU 设备无法容纳大规模模型参数时，便捷好用的分布式训练和推理需求就相继出现，业内也随之推出相应的工具。

基于 OneFlow 构建的 LiBai 模型库让分布式上手难度降到最低，用户不需要关注模型如何分配在不同的显卡设备，只需要修改几个配置数据就可以设置不同的分布式策略。当然，加速性能更是出众。

用 LiBai 搭建的 GLM 可以便捷地实现model parallel + pipeline parallel推理, 很好地解决单卡放不下大规模模型的问题。

那么，用户如何利用大规模模型训练与推理仓库 LiBai 来构建 GLM 的分布式推理部分？下面用一个小例子解释一下。

### 分布式推理具有天然优势

要知道，模型的参数其实就是许多 tensor，也就是以矩阵的形式出现，大模型的参数也就是大矩阵，并行策略就是把大矩阵分为多个小矩阵，并分配到不同的显卡或不同的设备上，基础的 LinearLayer 在LiBai中的实现代码如下：

```python
class Linear1D(nn.Module):
    def __init__(self, in_features, out_features, parallel="data", layer_idx=0, ...):
        super().__init__()

        if parallel == "col":
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(0)])
        elif parallel == "row":
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.split(1)])
        elif parallel == "data":
            weight_sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        else:
            raise KeyError(f"{parallel} is not supported! Only support ('data', 'row' and 'col')")

        self.weight = flow.nn.Parameter(
            flow.empty(
                (out_features, in_features),
                dtype=flow.float32,
                placement=dist.get_layer_placement(layer_idx),  # for pipeline parallelism placement
                sbp=weight_sbp,
            )
        )
        init_method(self.weight)
        ...
    
    def forward(self, x):
        ...
```

在这里，用户可选择去如何切分 Linear 层的矩阵，如何切分数据矩阵，而OneFlow 中的 SBP 控制竖着切、横着切以及其他拆分矩阵的方案（模型并行、数据并行），以及通过设置 Placement 来控制这个 LinearLayer 是放在第几张显卡上（流水并行）。

所以，根据 LiBai 中各种 layer 的设计原理以及基于 OneFlow 中 tensor 自带的 SBP 和 Placement 属性的天然优势，使得用户搭建的模型能够很简单地就实现数据并行、模型并行以及流水并行操作。

### GLM 推理的 Demo 演示

这里为用户展示 LiBai 中 GLM 便捷的4卡`model parallel+pipeline parallel`推理 Demo，模型可在 HuggingFace 上获取：https://huggingface.co/models?filter=glm


#### glm-10b的文件结构

```python
$ tree data
path/to/glm-10b
├── added_tokens.json
├── vocab.json
├── merges.txt
├── config.json
└── pytorch_model.bin
```

#### 推理

运行以下代码：
```bash
# 运行前修改 glm_inference.py 中 `pad_token_id=0, eos_token_id=50258, bos_token_id=50000`
python3 -m oneflow.distributed.launch --nproc_per_node 4 demo.py
```

```python
# model parallel + pipeline parallel demo

import oneflow as flow
from projects.GLM.tokenizer.glm_tokenizer import GLMGPT2Tokenizer
from libai.utils import distributed as dist
from projects.GLM.configs.glm_inference import cfg
from projects.GLM.modeling_glm import GLMForConditionalGeneration
from projects.GLM.utils.glm_loader import GLMLoaderHuggerFace
from omegaconf import DictConfig

# 只需简单配置并行方案
parallel_config = DictConfig(
    dict(
        data_parallel_size=1,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        pipeline_num_layers=2 * 24
    )
)
dist.setup_dist_util(parallel_config)

tokenizer = GLMGPT2Tokenizer.from_pretrained("/path/to/glm-10b")
input_ids = tokenizer.encode(
    [
        "Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai."
    ],
    return_tensors="of",
)
inputs = {"input_ids": input_ids, "attention_mask": flow.ones(input_ids.size())}
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)

sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
placement = dist.get_layer_placement(0)

loader = GLMLoaderHuggerFace(GLMForConditionalGeneration, cfg, "/path/to/glm-10b")
model = loader.load()

outputs = model.generate(
    inputs=inputs['input_ids'].to_global(sbp=sbp, placement=placement), 
    position_ids=inputs['position_ids'].to_global(sbp=sbp, placement=placement), 
    generation_attention_mask=inputs['generation_attention_mask'].to_global(sbp=sbp, placement=placement), 
    max_length=512
)
res = tokenizer.decode(outputs[0])
if dist.is_main_process():
    print(res)

>>> [CLS] Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.<|endoftext|> <|startofpiece|>  Stanford University and a co-founder of <|endofpiece|>

```

#### glm-10b-chinese的文件结构

```python
$ tree data
path/to/glm-10b-chinese
├── added_tokens.json
├── cog-pretrain.model
├── config.json
└── pytorch_model.bin
```

#### 推理

运行以下代码：
```bash
# 运行前修改 glm_inference.py 中 `pad_token_id=50000, eos_token_id=50007, bos_token_id=None`
python3 -m oneflow.distributed.launch --nproc_per_node 4 demo.py
```

```python
# model parallel + pipeline parallel demo

import oneflow as flow
from projects.GLM.tokenizer.glm_tokenizer import GLMChineseTokenzier
from libai.utils import distributed as dist
from projects.GLM.configs.glm_inference import cfg
from projects.GLM.modeling_glm import GLMForConditionalGeneration
from projects.GLM.utils.glm_loader import GLMLoaderHuggerFace
from omegaconf import DictConfig

# 只需简单配置并行方案
parallel_config = DictConfig(
    dict(
        data_parallel_size=1,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        pipeline_num_layers=2 * 24
    )
)
dist.setup_dist_util(parallel_config)

tokenizer = GLMChineseTokenzier.from_pretrained("/path/to/glm-10b-chinese")
input_ids = tokenizer.encode(
    [
        "凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。"
    ],
    return_tensors="of",
)
inputs = {"input_ids": input_ids, "attention_mask": flow.ones(input_ids.size())}
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)

sbp = dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
placement = dist.get_layer_placement(0)

loader = GLMLoaderHuggerFace(
    GLMForConditionalGeneration, 
    cfg, 
    "/path/to/glm-10b-chinese",
    embedding_dropout_prob=0,
    attention_dropout_prob=0,
    output_dropout_prob=0,
)
model = loader.load()

outputs = model.generate(
    inputs=inputs['input_ids'].to_global(sbp=sbp, placement=placement), 
    position_ids=inputs['position_ids'].to_global(sbp=sbp, placement=placement), 
    generation_attention_mask=inputs['generation_attention_mask'].to_global(sbp=sbp, placement=placement), 
    max_length=512
)

res = tokenizer.decode(outputs[0])
if dist.is_main_process():
    print(res)

>>> [CLS] 凯旋门位于意大利米兰市古城堡旁。1807年为纪念 [MASK] 而建,门高25米,顶上矗立两武士青铜古兵车铸像。 <|endoftext|> <|startofpiece|> 拿破仑军队攻克米兰城 <|endofpiece|>
```

#### 使用 One-GLM 训练的模型进行推理

LiBai对于OneFlow的模型加载同样方便，如果你希望使用one-glm训练后的模型进行推理，只需简单的将上述demo中的 GLMLoaderHuggerFace 替换为 GLMLoaderLiBai。