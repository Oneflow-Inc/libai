from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.data import build_nlp_test_loader, build_nlp_train_val_test_loader
from libai.data.datasets import GPT2Dataset
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

dataloader.train = LazyCall(build_nlp_train_val_test_loader)(
    dataset=[
        LazyCall(GPT2Dataset)(
            name="gpt-2",
            data_prefix="/workspace/data/libai_dataset/loss_compara_content_sentence",
            indexed_dataset=LazyCall(get_indexed_dataset)(
                data_prefix="/workspace/data/libai_dataset/loss_compara_content_sentence",
                data_impl="mmap",
                skip_warmup=False,
            ),
            max_seq_length=1024,
            seed=1234,
        ),
    ],
    train_val_test_num_samples=None,  # a hint for deferred assignment
    splits=[[949.0, 50.0, 1.0]],
    weights=[1.0],
    num_workers=4,
)
