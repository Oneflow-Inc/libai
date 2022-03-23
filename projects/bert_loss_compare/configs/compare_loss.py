from datetime import date
from libai.config import LazyCall, get_config

from libai.scheduler import WarmupMultiStepLR

model = get_config("common/models/bert.py").pretrain_model
graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim

train.eval_iter = 10

dataloader = dict()

# Set all dropout to 0.
model.cfg.hidden_dropout_prob = 0.0
model.cfg.attention_probs_dropout_prob = 0.0

# Set matched model arguments
model.cfg.hidden_layers = 5
model.cfg.hidden_size = 384
model.cfg.intermediate_size = 1536
model.cfg.num_attention_heads = 16
model.cfg.max_position_embeddings = 512

train.train_iter = 1000
train.micro_batch_size = 16
train.log_period = 20
train.warmup_ratio = 0.01
train.evaluation.enabled = False


# Set a constant lr scheduler after warmup
optim.lr = 0.0001
train.scheduler = LazyCall(WarmupMultiStepLR)(warmup_factor=0.1, milestones=[0.99])

data = dict(
    # Pad the vocab size to be divisible by this value
    # This is added for computational efficiency reasons.
    make_vocab_size_divisible_by=128,
    split="949,50,1",
    merge_file=None,
    vocab_extra_ids=0,
    seq_length=512,
    encoder_seq_length=None,
    decoder_seq_length=None,
    sample_rate=1.0,
    mask_prob=0.15,
    short_seq_prob=0.1,
    mmap_warmup=True,
    tokenizer_setup=True,
    # What type of tokenizer to use
    # "BertWordPieceLowerCase",
    # "BertWordPieceCase",
    # "GPT2BPETokenizer",
    # "BertCNWWMTokenizer",
    tokenizer_type="BertCNWWMTokenizer",
    # What type of dataset to use
    dataset_type="standard_bert",
    # Implementation of indexed datasets, choose from `lazy`, `cached`, `mmap`, `infer`.
    data_impl="mmap",
    reset_position_ids=False,
    reset_attention_mask=False,
    eod_mask_loss=False,
    use_external_dataset=False,
    # Dataloader type and number of workers
    dataloader_type="single",
    num_workers=4,
)


today = date.today()
train.output_dir = f"loss_align/bert_loss_compare/{today}"
