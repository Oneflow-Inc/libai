from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.data import build_nlp_test_loader, build_nlp_train_val_test_loader
from libai.data.datasets import RobertaDataset
from libai.data.data_utils import get_indexed_dataset

from libai.tokenizer import RobertaTokenizer


tokenization = OmegaConf.create()

tokenization.tokenizer = LazyCall(RobertaTokenizer)(
    vocab_file="roberta-vocab.json", merges_file="roberta-merges.txt"
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_nlp_train_val_test_loader)(
    dataset=[
        LazyCall(RobertaDataset)(
            name="roberta",
            data_prefix="/workspace/data/libai_dataset/loss_compara_content_sentence",
            indexed_dataset=LazyCall(get_indexed_dataset)(
                data_prefix="/workspace/data/libai_dataset/loss_compara_content_sentence",
                data_impl="mmap",
                skip_warmup=False,
            ),
            max_seq_length=512,
            mask_lm_prob=0.15,
            short_seq_prob=0.0,
            seed=1234,
            masking_style="bert",
        ),
    ],
    train_val_test_num_samples=None,  # a hint for deferred assignment
    splits=[[949.0, 50.0, 1.0]],
    weights=[1.0],
    num_workers=4,
)
