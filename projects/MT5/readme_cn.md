# MT5 模型文档

## 环境配置

1. 安装最新的 [OneFlow](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package)，支持 `Python` 版本 `3.7, 3.8, 3.9, 3.10`。

```bash
python3 -m pip install --upgrade pip #--user
```

```bash
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/[PLATFORM]
```

`[PLATFORM]` 根据平台设置，目前支持：`cu112`, `cu102` 和 `cpu`

2. 从 github 获取 [Libai](https://github.com/Oneflow-Inc/libai) 库

```bash
git clone --recursive https://github.com/Oneflow-Inc/libai.git
```

或

```bash
git clone --recursive git@github.com:Oneflow-Inc/libai.git
```

然后进入到 `libai` 工程目录下并切换到 `dev_optimize_MT5` 分支


```bash
cd libai
git checkout dev_optimize_MT5
```

3. 安装 Libai

在 `libai` 根目录下运行以下命令：

```bash
pip install -r requirements.txt # 安装依赖项
pip install -e . # 以可编辑模式安装 Libai，这样对于 Libai 库核心功能的修改都可以直接生效
```

## MT5 训练流程

MT5模型目录: `PATH_TO_LIBAI_ROOT/projects/MT5`

### 下载demo训练数据

在 `libai` 项目的根目录下运行以下命令:

```bash
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/bert-base-chinese-vocab.txt -P ./data_test/bert_data/
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.bin -P ./data_test/bert_data/
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.idx -P ./data_test/bert_data/
```

如果是准备自己的数据集可以参考文档：`https://libai.readthedocs.io/en/latest/tutorials/basics/Preprocessing_Dataset.html`

### 运行单机单卡训练

在 `libai` 项目的根目录下运行以下命令：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/train.sh tools/train_net.py projects/MT5/configs/mt5_pretrain.py 1
```

A100 测试数据：
```bash
time: ~0.23 s/iter 
total_throughput: ~70 samples/s
```

环境变量的含义：

- `NODE`: 集群中的机器数
- `NODE_RANK`: 当前节点的序号
- `ADDR`: master 节点的 ip 地址
- `PORT`: master 节点的端口号


### 加载 HuggingFace 预训练模型

以 [mt5-base](https://huggingface.co/google/mt5-base/tree/main) 为例。
 
在 `libai` 项目根目录运行以下命令下载预训练模型：

```bash
wget https://huggingface.co/google/mt5-base/resolve/main/pytorch_model.bin -P ./data_test
wget https://huggingface.co/google/mt5-base/raw/main/config.json -P data_test/
```

修改 `PATH_TO_LIBAI_ROOT/projects/MT5/configs/mt5_pretrain.py` 文件 `13` 行如下：

```python3
pretrained_model_path = "./data_test"
```

即可加载 HuggingFace 的预训练模型继续微调：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/train.sh tools/train_net.py projects/MT5/configs/mt5_pretrain.py 1
```

注：加载 `HuggingFace` 的模型需要安装 `PyTorch`

### 运行各种并行模式配置

#### 运行单机8卡数据并行训练
在 `libai` 项目的根目录下运行以下命令：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/train.sh tools/train_net.py projects/MT5/configs/mt5_pretrain.py 8
```

A100 测试数据：
```bash
time: ~0.3 s/iter
total_throughput: ~426 samples/s
```

#### 运行单机8卡朴素流水并行

修改 `PATH_TO_LIBAI_ROOT/projects/MT5/configs/mt5_pretrain.py` 文件 `41~43` 行如下：

```python3
dist=dict(
    data_parallel_size=1,
    tensor_parallel_size=1,
    pipeline_parallel_size=8,
    pipeline_num_layers=2 * model.cfg.hidden_layers,
),
```

在 `libai` 项目的根目录下运行以下命令：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/train.sh tools/train_net.py projects/MT5/configs/mt5_pretrain.py 8
```

A100 测试数据：
```bash
time: ~0.73 s/iter
total_throughput: ~22 samples/s
```

#### 运行单机8卡3D混合并行(数据并行度2 + 张量并行度2 + 朴素流水并行度2)

修改 `PATH_TO_LIBAI_ROOT/projects/MT5/configs/mt5_pretrain.py` 文件 `41~43` 行如下：

```python3
dist=dict(
    data_parallel_size=2,
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    pipeline_num_layers=2 * model.cfg.hidden_layers,
),
```

在 `libai` 项目的根目录下运行以下命令：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/train.sh tools/train_net.py projects/MT5/configs/mt5_pretrain.py 8
```

A100 测试数据：
```bash
time: ~0.34 s/iter
total_throughput: ~93 samples/s
```

#### 运行2机4卡3D混合并行(数据并行度2 + 张量并行度2 + 朴素流水并行度2)

修改 `PATH_TO_LIBAI_ROOT/projects/MT5/configs/mt5_pretrain.py` 文件 `41~43` 行如下：

```python3
dist=dict(
    data_parallel_size=2,
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    pipeline_num_layers=2 * model.cfg.hidden_layers,
),
```

在0号机器 `libai` 项目的根目录下运行以下命令：

```bash
NODE=2 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/train.sh tools/train_net.py projects/MT5/configs/mt5_pretrain.py 4
```

在1号机器 `libai` 项目的根目录下运行以下命令：

```bash
NODE=2 NODE_RANK=1 ADDR=192.168.30.21 PORT=12345 bash tools/train.sh tools/train_net.py projects/MT5/configs/mt5_pretrain.py 4
```

## MT5 推理流程

### 单卡推理

#### 加载 HuggingFace 预训练模型推理

以 [mt5-base](https://huggingface.co/google/mt5-base/tree/main) 为例。
 
在 `libai` 项目根目录运行以下命令下载预训练模型：

```bash
wget https://huggingface.co/google/mt5-base/resolve/main/pytorch_model.bin -P ./data_test
wget https://huggingface.co/google/mt5-base/raw/main/config.json -P data_test/
```

下载词表文件:

```bash
wget https://huggingface.co/google/mt5-base/resolve/main/spiece.model -P data_test/
```

修改 `libai/inference/text_generation.py` 文件，第 `24` 行为:

```python3
tokenizer = T5Tokenizer(
    "data_test/spiece.model",
    add_bos_token=True,
)
```

第 `98~99` 行为：

```python3
model_path="data_test/",
mode="huggingface", 
```

然后在 `libai` 项目的根目录下运行以下命令：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/infer.sh libai/inference/text_generation.py 1
```

#### 加载 Libai 预训练模型推理

修改 `libai/inference/text_generation.py` 文件，第 `98~99` 行为：

```python3
model_path="projects/MT5/output/mt5_output/model_final/model",
mode="libai", 
```

然后在 `libai` 项目的根目录下运行以下命令：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/infer.sh libai/inference/text_generation.py 1
```


### 2卡推理，模型分段并行


修改 `libai/inference/text_generation.py` 文件，第 `93~95` 行为:

```python3
data_parallel=1,
tensor_parallel=1,
pipeline_parallel=2,
```

在 `libai` 项目的根目录下运行以下命令：

```bash
NODE=1 NODE_RANK=0 ADDR=192.168.30.21 PORT=12345 bash tools/infer.sh libai/inference/text_generation.py 2
```


