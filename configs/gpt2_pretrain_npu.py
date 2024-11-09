from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
# from .common.models.gpt import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.gpt_dataset import dataloader, tokenization

# from .common.models.graph_npu import graph
from omegaconf import DictConfig
from libai.models import GPTForPreTraining

vocab_file = "/home/nvme1n1/home/data/gpt2/gpt2-vocab.json"
merge_files = "/home/nvme1n1/home/data/gpt2/gpt2-merges.txt"
data_prefix = "/home/nvme1n1/home/data/gpt2/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merge_files
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
dataloader.test[0].dataset.data_prefix = data_prefix
dataloader.test[0].dataset.indexed_dataset.data_prefix = data_prefix


from libai.models.utils import GraphBase
graph = dict(
    # options for graph or eager mode
    enabled=False,
    debug=3,  # debug mode for graph
    # =========== optimization settings on Graph mode, default:True ===========
    allow_fuse_add_to_output=False,
    allow_fuse_model_update_ops=False, # try to fuse cast + scale + l1_l2_regularize_gradient + model_update to one op to improve performance.
    allow_fuse_cast_scale=False, # try to fuse cast and scalar_mul_by_tensor to improve performance.
    enable_cudnn_conv_heuristic_search_algo=False,
    enable_compress_memory=False,
    enable_choose_best_memory_allocation=False,
    enable_auto_parallel=False,
    enable_multi_tensor_update=False,     # merge small optimizer kernels to reduce kernel launch overhead
    enable_fused_model_update_cast=False, # only works in AMP Mode, 
    # =========== optimization settings on Graph mode, default:True ===========
    auto_parallel=dict(
        enabled=False,
        enable_auto_parallel_ignore_user_sbp_config=False,  # ignore all .to_global() in graph
        trunk_algo=True,  # consider overlapping calculate time and transfer time
        sbp_collector=False,  # use proxy node when one node transfer to many nodes
    ),
    train_graph=LazyCall(GraphBase)(
        is_train=True,
    ),
    global_mode=dict(
        enabled=False,
    ),
    eval_graph=LazyCall(GraphBase)(is_train=False),
)

graph = DictConfig(graph)


cfg = dict(
    hidden_layers=6,
    vocab_size=30522,
    hidden_size=384,
    ffn_hidden_size=1536,
    num_attention_heads=12,
    max_seq_length=1024,
    embedding_dropout_prob=0,
    attention_dropout_prob=0,
    output_dropout_prob=0,
    layernorm_epsilon=1e-5,
    initializer_range=0.02,
    use_scaled_init_for_output_weights=True,
    bias_gelu_fusion=False,  # need op/kernel: fused_bias_add_gelu、fused_bias_add_gelu_grad
    bias_dropout_fusion=False,
    scale_mask_softmax_fusion=False,  # need op/kernel: fused_tril_scale_softmax_mask_scale、fused_tril_scale_softmax_mask_scale_grad
    apply_query_key_layer_scaling=True,
    apply_residual_post_layernorm=False,
    amp_enabled=False,
)


cfg = DictConfig(cfg)
model = LazyCall(GPTForPreTraining)(cfg=cfg)

model.cfg.embedding_dropout_prob = 0
model.cfg.attention_dropout_prob = 0

# 参数量1.1B
# model.cfg.num_attention_heads = 25
# model.cfg.hidden_size = 1600
# model.cfg.ffn_hidden_size = 1280 * 4
# model.cfg.hidden_layers = 40
# model.cfg.max_seq_length = 1024

# libai default config
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 384
model.cfg.ffn_hidden_size = 1536
model.cfg.hidden_layers = 6
model.cfg.max_seq_length = 1024

# # GPT-2 small(参数量117M)
# model.cfg.num_attention_heads = 12
# model.cfg.hidden_size = 768
# model.cfg.ffn_hidden_size = 3072
# model.cfg.hidden_layers = 12
# model.cfg.max_seq_length = 1024

# # benchmark:https://libai.readthedocs.io/en/latest/tutorials/get_started/Benchmark.html#data-parallel
# GPT-2 medium（参数量345M)
# model.cfg.num_attention_heads = 16
# model.cfg.hidden_size = 1024
# model.cfg.ffn_hidden_size = 4096
# model.cfg.hidden_layers = 24
# model.cfg.max_seq_length = 1024

# GPT-2 large(参数量774M)
# model.cfg.num_attention_heads = 20
# model.cfg.hidden_size = 1280
# model.cfg.ffn_hidden_size = 5120
# model.cfg.hidden_layers = 36
# model.cfg.max_seq_length = 1024

train.train_micro_batch_size = 6

train.input_placement_device = "cpu"
# train.input_placement_device = "npu"

train.dist.pipeline_num_layers = model.cfg.hidden_layers

for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_seq_length

optim.lr = 1.5e-4

train.train_micro_batch_size = 4
train.amp.enabled = False

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "./output/gpt2_npu_output"
