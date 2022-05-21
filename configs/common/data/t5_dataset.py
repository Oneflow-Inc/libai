from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.data import build_nlp_test_loader, build_nlp_train_val_test_loader
from libai.data.datasets import T5Dataset
from libai.data.data_utils import get_indexed_dataset

from libai.tokenizer import BertTokenizer


tokenization = OmegaConf.create()

tokenization.setup = True

special_tokens = []
for i in range(100):
    special_tokens.append(f"<extra_id_{i}>")
tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="/workspace/data/libai_dataset/bert-base-chinese-vocab.txt",
    do_lower_case=True,
    do_chinese_wwm=True,
    bos_token="[BOS]",
    eos_token="[EOS]",
    additional_special_tokens=special_tokens,
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_nlp_train_val_test_loader)(
    dataset=[
        LazyCall(T5Dataset)(
            name="t5",
            data_prefix="/workspace/data/libai_dataset/loss_compara_content_sentence",
            indexed_dataset=LazyCall(get_indexed_dataset)(
                data_prefix="/workspace/data/libai_dataset/" "/loss_compara_content_sentence",
                data_impl="mmap",
                skip_warmup=False,
            ),
            max_seq_length=512,
            max_seq_length_dec=128,
            masked_lm_prob=0.15,
            short_seq_prob=0.1,
            seed=1234,
        ),
    ],
    train_val_test_num_samples=None,  # a hint for deferred assignment
    splits=[[949.0, 50.0, 1.0]],
    weights=[1.0],
    num_workers=4,
)
