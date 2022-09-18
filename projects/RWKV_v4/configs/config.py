import os

os.environ['VOCAB_SIZE'] = '50277'
os.environ['USE_L2_REG'] = '1'

os.environ['RWKV_FLOAT_MODE'] = 'bf16' # tf32 fp32 fp16 bf16
print(os.environ['RWKV_FLOAT_MODE'])

if os.environ['RWKV_FLOAT_MODE'] == 'fp32':
    os.environ['ONEFLOW_EP_CUDA_ENABLE_TF32_EXECUTION'] = '0'
else:
    os.environ['ONEFLOW_EP_CUDA_ENABLE_TF32_EXECUTION'] = '1'

from omegaconf import OmegaConf
import oneflow as flow
import libai

from libai.config import get_config
from libai.config import LazyCall
from libai.tokenizer import GPT2Tokenizer
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader

from projects.RWKV_v4.modeling.model import GPT, GPTConfig
from projects.RWKV_v4.utils.config_optimizer import get_RWKV_v4_config_optim
from projects.RWKV_v4.dataset import RWKVDataset

graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train
optim = LazyCall(flow.optim.Adam)(
    params=LazyCall(get_RWKV_v4_config_optim)(),
    lr=6e-4,
    betas = (0.9, 0.99),
    eps = 1e-8
)
# optim = LazyCall(flow.optim.SGD)(
#     params=LazyCall(get_RWKV_v4_config_optim)(grad_clip = 1.0),
#     lr=0.000001
# )
train.scheduler = LazyCall(libai.scheduler.WarmupExponentialLR)(max_iter=10000000, gamma=1.0, warmup_factor=0, warmup_iter=0)

load_torch_checkpoint = OmegaConf.create()
load_torch_checkpoint.enable = True
load_torch_checkpoint.weight_style = "pytorch"

# model = LazyCall(GPT)(vocab_size=int(os.environ['VOCAB_SIZE']), ctx_len=1024, model_type="RWKV", n_layer=12, n_embd=768)
# load_torch_checkpoint.path = "/fsx/BlinkDL/CODE/FP16/out_100a/all-0.pth"
# model = LazyCall(GPT)(vocab_size=int(os.environ['VOCAB_SIZE']), ctx_len=1024, model_type="RWKV", n_layer=24, n_embd=1024)
# load_torch_checkpoint.path = "/fsx/BlinkDL/CODE/FP16/out_400mbf16/all-8066.pth"
model = LazyCall(GPT)(vocab_size=int(os.environ['VOCAB_SIZE']), ctx_len=1024, model_type="RWKV", n_layer=24, n_embd=2048)
load_torch_checkpoint.path = "/fsx/BlinkDL/CODE/FP16/out_1b2/all-81.pth"

train.train_micro_batch_size = 1
train.train_iter = 0
train.train_epoch = 1
train.log_period = 1

if os.environ['RWKV_FLOAT_MODE'] == 'bf16':
    train.amp.enabled = True
    train.amp.type = "bf16"
elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
    train.amp.enabled = True
    train.amp.type = "fp16"
else:
    train.amp.enabled = False
    train.amp.type = "fp32"

train.rdma_enabled = False
train.activation_checkpoint.enabled = False

train.input_placement_device = "cpu"
train.dist.data_parallel_size = 1
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1
train.dist.pipeline_num_layers = model.n_layer

train.zero_optimization.enabled = True
train.zero_optimization.stage = 2

n_gpu = train.dist.data_parallel_size * train.dist.tensor_parallel_size * train.dist.pipeline_parallel_size
batch_sz = train.train_micro_batch_size * n_gpu

train.output_dir = "rwkv_out/"
# train.load_weight = "output/RWKV_v4/model/model_final/"

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)( # must turn off Shuffle !!!
    dataset=[
        LazyCall(RWKVDataset)(
            # datacfg=['txt', "enwik8"],
            # datacfg=['numpy', "train.npy"],
            datacfg=['idx_bin', "/fsx/BlinkDL/pile/pile_20B_tokenizer_text_document"],
            ctx_len=1024,
            idx_begin=1
        ),
    ],
    num_workers=1, # must be 1
)
