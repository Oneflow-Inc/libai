from omegaconf import OmegaConf

from libai.config import get_config
from libai.config import LazyCall
from libai.tokenizer import GPT2Tokenizer

from projects.RWKV_v4.modeling.model import GPT, GPTConfig
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from projects.RWKV_v4.dataset import RWKVDataset

graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim

model = LazyCall(GPT)(vocab_size=6064, ctx_len=1024, model_type="RWKV", n_layer=6, n_embd=512)

datafile = "path/to/data/enwik8"
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

train.train_micro_batch_size = 12
train.train_iter = 0
train.train_epoch = 1

train.rdma_enabled = False
train.activation_checkpoint.enabled = True

train.input_placement_device = "cpu"
train.dist.data_parallel_size = 1
train.dist.tensor_parallel_size = 2
train.dist.pipeline_parallel_size = 1
train.dist.pipeline_num_layers = model.n_layer

train.output_dir = "output/RWKV4/"
# train.load_weight = "output/RWKV_v4/model/model_final/"
