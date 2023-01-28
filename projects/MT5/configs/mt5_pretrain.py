from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.scheduler import WarmupExponentialLR
from configs.common.train import train
from configs.common.data.t5_dataset import dataloader, tokenization
from configs.common.models.graph import graph
from configs.common.optim import optim
from projects.MT5.configs.mt5_base import pretrain_model as model


vocab_file = "./data_test/bert_data/bert-base-chinese-vocab.txt"
data_prefix = "./data_test/bert_data/loss_compara_content_sentence"
pretrained_model_path = None

tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

# model config
model.cfg.model_type = "mt5"
model.cfg.hidden_dropout_prob = 0.1
model.cfg.attention_probs_dropout_prob = 0.1
model.cfg.embedding_dropout_prob = 0.1
model.cfg.padding_idx = 0
model.cfg.tie_word_embeddings = False
model.cfg.amp_enabled = True
model.cfg.initializer_range = 0.02
model.cfg.pretrained_model_path = None

train.update(
    dict(
        output_dir="projects/MT5/output/mt5_output",
        train_micro_batch_size=4,
        train_epoch=1,
        train_iter=24000,
        log_period=10,
        amp=dict(enabled=True),
        warmup_ratio=1 / 24,
        checkpointer=dict(period=5000, max_to_keep=20),
        input_placement_device="cpu",
        dist=dict(
            data_parallel_size=8,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            pipeline_num_layers=2 * model.cfg.hidden_layers,
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
        ),
    )
)

train.zero_optimization.enabled = False
train.zero_optimization.stage = 2
train.activation_checkpoint.enabled = False
train.num_accumulation_steps = 8
