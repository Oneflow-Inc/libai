from libai.data.build import build_nlp_train_loader
from omegaconf import OmegaConf

from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.scheduler import WarmupExponentialLR

from configs.common.train import train
from configs.common.models.graph import graph

from projects.T5.configs.optim import optim
from projects.T5.datasets.dataset import UnsuperviseT5Dataset, collate_fn
from projects.T5.configs.t5_config import pretrain_model as model


train_data_path = "/home/xiezipeng/libai/projects/T5/data/wudao_180g_test_bert_tokenized_512_train/part_0"


# dataloader
dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(UnsuperviseT5Dataset)(
            data_path=train_data_path,
        )
    ],
    collate_fn = collate_fn(
        vocab_size=12902,
        max_seq_length=512,
        noise_density=0.15,
        mean_noise_span_length=3,
        eos_token_id=12801,
        pad_token_id=0,
        decoder_start_token_id=12800
    )    
)

# model config
model.cfg.vocab_size = 12902
model.cfg.hidden_size = 512
model.cfg.hidden_layers = 8
model.cfg.num_attention_heads = 6
model.cfg.head_size = 64
model.cfg.intermediate_size = 1024
model.cfg.layernorm_eps = 1e-6
model.cfg.mlp_type = 'mt5'

train.dist.pipeline_num_layers = 2 * model.cfg.hidden_layers

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.update(
    dict(
        output_dir="./output/t5_output",
        train_micro_batch_size=8,
        train_epoch=1,
        train_iter=1000,
        log_period=10,
        amp=dict(enabled=True),
        warmup_ratio=1/24,
        dist=dict(
            data_parallel_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),

        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.001,
            gamma=1.0,
            warmup_method="linear",
            warmup_iter=0.0,
        ),
    )
)
