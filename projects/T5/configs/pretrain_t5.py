from random import sample
from libai import evaluation
from libai.data.build import build_nlp_train_loader
from omegaconf import OmegaConf

from libai.config import LazyCall
from libai.evaluation import PPLEvaluator, evaluator
from libai.scheduler import WarmupExponentialLR

from configs.common.train import train
from configs.common.models.graph import graph

from projects.T5.configs.optim import optim
from projects.T5.datasets.dataset import UnsuperviseT5Dataset, collate_fn
from projects.T5.configs.t5_config import pretrain_model as model


train_data_path = "/home/xiezipeng/libai/projects/T5/data/wudao_180g_test_bert_tokenized_512_train/part_0"
pretrained_model_path = "/home/xiezipeng/libai/projects/T5/data/pretrained-t5/randeng_t5_char_57M"

micro_batch_size = 64

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
model.cfg.bias_gelu_fusion = False
model.cfg.bias_dropout_fusion = False
model.cfg.apply_query_key_layer_scaling = False
model.cfg.mlp_type = 'mt5'
model.cfg.pretrained_model_path = pretrained_model_path

train.update(
    dict(
        output_dir="./output/t5_output",
        train_micro_batch_size=micro_batch_size,
        train_epoch=1,
        train_iter=240000,
        log_period=10,
        amp=dict(enabled=True),
        warmup_ratio=1/24,
        dist=dict(
            data_parallel_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            pipeline_num_layers = 2 * model.cfg.hidden_layers
        ),

        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.001,
            gamma=1.0,
            warmup_method="linear",
            warmup_iter=0.0,
        ),

        evaluation=dict(
            evaluator=LazyCall(PPLEvaluator)(),
            enabled=True,
            eval_iter=1e5,
            eval_period=5000,
        )
    )
)