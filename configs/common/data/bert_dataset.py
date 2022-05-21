from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.data import build_nlp_test_loader, build_nlp_train_val_test_loader
from libai.data.datasets import BertDataset
from libai.data.data_utils import get_indexed_dataset

from libai.tokenizer import BertTokenizer


tokenization = OmegaConf.create()

tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="bert-base-chinese-vocab.txt",
    do_lower_case=True,
    do_chinese_wwm=True,
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_nlp_train_val_test_loader)(
    dataset=[
        LazyCall(BertDataset)(
            name="bert",
            data_prefix="/workspace/data/libai_dataset/loss_compara_content_sentence",
            indexed_dataset=LazyCall(get_indexed_dataset)(
                data_prefix="/workspace/data/libai_dataset/loss_compara_content_sentence",
                data_impl="mmap",
                skip_warmup=False,
            ),
            max_seq_length=512,
            mask_lm_prob=0.15,
            short_seq_prob=0.1,
            binary_head=True,
            seed=1234,
            masking_style="bert-cn-wwm",
        ),
    ],
    train_val_test_num_samples=None,  # a hint for deferred assignment
    splits=[[949.0, 50.0, 1.0]],
    weights=[1.0],
    num_workers=4,
)
