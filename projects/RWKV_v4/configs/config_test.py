from omegaconf import OmegaConf

from libai.config import get_config
from libai.config import LazyCall
from libai.tokenizer import GPT2Tokenizer
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
import oneflow as flow

from projects.RWKV_v4.modeling.model import GPT, GPTConfig
from projects.RWKV_v4.dataset import RWKVDataset
from projects.RWKV_v4.utils.config_optimizer import get_RWKV_v4_config_optim


load_torch_checkpoint = OmegaConf.create()
load_torch_checkpoint.enable = True
load_torch_checkpoint.weight_style = "pytorch"
load_torch_checkpoint.path = "data_test/rwkv_test/for_load.pth"

graph = get_config("common/models/graph.py").graph

graph.enabled = True

optim = LazyCall(flow.optim.Adam)(
    params=LazyCall(get_RWKV_v4_config_optim)(),
    lr=8e-4,
)

model = LazyCall(GPT)(
    vocab_size=6064, 
    ctx_len=1024, 
    model_type="RWKV", 
    n_layer=6, 
    n_embd=512
)

train = get_config("common/train.py").train
train.input_placement_device = "cpu"
train.dist.pipeline_num_layers = 6
train.train_micro_batch_size = 4
train.scheduler = LazyCall(flow.optim.lr_scheduler.StepLR)(
    step_size=1000, 
    gamma=1.0
)

train.amp.enabled = True
train.amp.type = "bf16"

datafile = "data_test/rwkv_test/enwik8"

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

train.train_iter=300
# train.train_epoch = 1

train.output_dir = "output/rwkv_output_loss_compare"
train.rdma_enabled = False
train.evaluation.enabled = False
train.activation_checkpoint.enabled = True
