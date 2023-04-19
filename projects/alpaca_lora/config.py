from libai.config import LazyCall
from omegaconf import OmegaConf
# from libai.evaluation import PPLEvaluator
from projects.alpaca_lora.modeling import AlpacaModel
from projects.alpaca_lora.dataset import LoraDataset
from configs.common.optim import optim
from libai.data import build_nlp_train_loader
from libai.scheduler import WarmupExponentialLR
from configs.common.train import train
from configs.common.models.graph import graph

graph.global_mode.enabled = True

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(LoraDataset)(
            data_path="yahma/alpaca-cleaned",
            prompt_dir="data_test/lora_data/templates",
            prompt_template_name="alpaca",
            tokenizer_name="decapoda-research/llama-7b-hf",
            max_length=256,
        ),
    ],
#     train_val_test_num_samples=None,  # a hint for deferred assignment
#     splits=[[949.0, 50.0, 1.0]],
#     weights=[1.0],
#     num_workers=4,
)

model = LazyCall(AlpacaModel)(
    model_name="HuggingFaceM4/tiny-random-LlamaForCausalLM",
    lora_r=8,
    lora_alpha=16,
    lora_target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    lora_bias="none",
    lora_task_type="CAUSAL_LM",
)

train.input_placement_device = "cuda"


# train.dist.pipeline_num_layers = 12
optim.lr = 3e-4

train.update(
    dict(
        output_dir="output/alpaca_lora_output",
        train_micro_batch_size=4,
        # test_micro_batch_size=4,
        # train_epoch=33,
        train_iter=100,
        log_period=10,
        amp=dict(enabled=False),
        warmup_ratio=0,
        checkpointer=dict(period=8000, max_to_keep=20),
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            # pipeline_num_layers=model.cfg.hidden_layers,
        ),
        scheduler=LazyCall(WarmupExponentialLR)(
            warmup_factor=0.0,
            gamma=1.0,
            warmup_method="linear",
            warmup_iter=0.0,
        ),
        evaluation=dict(
            enabled=False,
            # evaluator=LazyCall(PPLEvaluator)(),
            # eval_iter=250,
            # eval_period=4000,
        ),
        rdma_enabled=False,
    )
)