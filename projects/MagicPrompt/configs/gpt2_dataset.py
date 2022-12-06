from libai.config import LazyCall
from libai.data import build_nlp_train_val_test_loader
from configs.common.data.gpt_dataset import tokenization, dataloader
from libai.tokenizer import GPT2Tokenizer
from libai.data.datasets import GPT2Dataset
from libai.data.data_utils import get_indexed_dataset


data_prefix = "/data/home/magicprompt/train/en_train_mmap_text_sentence"

tokenization.tokenizer = LazyCall(GPT2Tokenizer)(
    vocab_file="/data/home/magicprompt/vocab.json",
    merges_file="/data/home/magicprompt/merges.txt",
    do_lower_case=True,
    do_chinese_wwm=True,
)
tokenization.append_eod = False

dataloader.train = LazyCall(build_nlp_train_val_test_loader)(
    dataset=[
        LazyCall(GPT2Dataset)(
            name="gpt-2",
            data_prefix=data_prefix,
            indexed_dataset=LazyCall(get_indexed_dataset)(
                data_prefix=data_prefix,
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
