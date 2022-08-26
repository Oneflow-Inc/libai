from omegaconf import OmegaConf

from libai.config import get_config
from libai.config import LazyCall
from libai.tokenizer import GPT2Tokenizer
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
import oneflow as flow

from projects.RWKV_v4.modeling.model import GPT, GPTConfig
from projects.RWKV_v4.dataset import RWKVDataset
from projects.RWKV_v4.utils.config_optimizer import get_RWKV_v4_config_optim


test = OmegaConf.create()
test.enable = True
test.weight_style = "pytorch"
test.path = "/home/zhangxiaoyu/RWKV-LM/RWKV-v4/for_load.pth"

graph = get_config("common/models/graph.py").graph

graph.enabled = True

# optim = get_config("common/optim.py").optim
optim = LazyCall(flow.optim.Adam)(
    params=LazyCall(get_RWKV_v4_config_optim)(),
    lr=8e-4,
)


# 配置model
model = LazyCall(GPT)(vocab_size=6064, ctx_len=1024, model_type="RWKV", n_layer=6, n_embd=512)

# 训练过程
train = get_config("common/train.py").train
train.input_placement_device = "cpu"
train.dist.pipeline_num_layers = 6
train.train_micro_batch_size = 4
train.scheduler = LazyCall(flow.optim.lr_scheduler.StepLR)(step_size=1000, gamma=1.0)

# false = fp32
train.amp.enabled = True

datafile = "/home/zhangxiaoyu/RWKV-LM/data/enwik8"
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
    num_workers=4,
)

# train.train_iter=3
train.train_epoch = 1

train.output_dir = "output/rwkv_output_loss_compare"
# train.load_weight = "/home/zhangxiaoyu/RWKV-LM/libai/projects/RWKV_v4/model/output_model/"
train.rdma_enabled = False

# model.cfg.hidden_dropout_prob= 0.0 # 关闭所有的dropout
# model.cfg.attention_probs_dropout_prob= 0.0
# model.cfg.bias_dropout_fusion= False

# train.dist.pipeline_parallel_size=2
train.evaluation.enabled = False

# train.train_iter=5
# train.dist.tensor_parallel_size = 2  # 并行度为 4 的模型并行
# train.dist.tensor_parallel_size = 4  # 并行度为 4 的模型并行
train.activation_checkpoint.enabled = True
