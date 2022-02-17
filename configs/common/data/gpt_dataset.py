from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.data import build_nlp_test_loader, build_nlp_train_val_test_loader
from libai.data.build import build_megatron_gpt_train_val_test_loader
from libai.data.datasets.megatron_gpt_dataset import build_train_valid_test_datasets
from libai.data.data_utils import get_indexed_dataset

from libai.tokenizer import GPT2Tokenizer


tokenization = OmegaConf.create()

tokenization.tokenizer = LazyCall(GPT2Tokenizer)(
    vocab_file="/workspace/data/gpt_dataset/gpt2-vocab.json",
    merges_file="/workspace/data/gpt_dataset/gpt2-merges.txt",
    do_lower_case=True,
    do_chinese_wwm=True,
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_megatron_gpt_train_val_test_loader)(
    dataset=OmegaConf.create(),
    train_val_test_datasets=LazyCall(build_train_valid_test_datasets)(
        data_prefix=["/workspace/data/libai_dataset/loss_compara_content_sentence"],
        data_impl="mmap",
        splits_string="949,50,1",
        train_valid_test_num_samples=[2000000, 20040, 40],
        seq_length=1024,
        seed=1234,
        skip_warmup=True,
    ),
    seed=1234,
    num_workers=0,
)
