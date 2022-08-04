from omegaconf import OmegaConf

from libai.config import get_config
from libai.config import LazyCall
from libai.tokenizer import GPT2Tokenizer

# 配置 model
from projects.RWKV_V4.modeling.model import GPT ,GPTConfig
# 配置 dataloader `build_image_train_loader` 和 `build_image_test_loader` 是 LiBai 提供的用于创建图像数据的训练集和测试集 DataLoader 的两个函数
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
# 导入自定义的 dataset
from projects.RWKV_V4.dataset import RWKVDataset

graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim

# 配置model
model=LazyCall(GPT)(
    vocab_size=6064,
    ctx_len=1024,
    model_type='RWKV',
    n_layer=6,
    n_embd=512
)

# 训练过程
train = get_config("common/train.py").train
train.input_placement_device = "cpu"
train.dist.pipeline_num_layers = 6
train.train_micro_batch_size = 12

datafile="/home/chenqiaoling/RWKV-LM/data/enwik8"
# 获得一个 DataLoader 的配置对象
dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(RWKVDataset)(
            data_dir=datafile,
            ctx_len=1024,
            epoch_length_fixed=9996,
        ),
    ],
    num_workers=1,
)

train.train_iter=0
train.train_epoch=1

train.output_dir = "output/rwkv_output_loss_compare"
train.load_weight = "/home/chenqiaoling/RWKV-LM/libai/projects/RWKV_V4/model/output_model/" # 采用同一个model进行初始化
train.rdma_enabled = False

# model.cfg.hidden_dropout_prob= 0.0 # 关闭所有的dropout
# model.cfg.attention_probs_dropout_prob= 0.0
# model.cfg.bias_dropout_fusion= False

train.dist.pipeline_parallel_size=2
train.evaluation.enabled = False
# train.dist.tensor_parallel_size = 4  # 并行度为 4 的模型并行
