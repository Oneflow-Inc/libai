from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.data import BertDataset, build_nlp_test_loader, build_nlp_train_val_test_loader
from libai.data.data_utils import get_indexed_dataset

from libai.tokenizer import BertTokenizer


tokenization = OmegaConf.create()

tokenization.setup = True
tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="/workspace/idea_model/idea_bert/bert-base-chinese-vocab.txt",
    do_lower_case=True,
)
tokenization.append_eod = False
tokenization.make_vocab_size_divisible_by = 128

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_nlp_train_val_test_loader)(
    dataset=[
        LazyCall(BertDataset)(
            data_prefix="/workspace/idea_model/idea_bert/output_data/loss_compara_content_sentence",
            indexed_dataset=LazyCall(get_indexed_dataset)(
                data_prefix="/workspace/idea_model/idea_bert/output_data"
                "/loss_compara_content_sentence",
                data_impl="mmap",
                skip_warmup=False,
            ),
            max_seq_length=512,
            mask_lm_prob=0.15,
            short_seq_prob=0.1,
        ),
    ],
    splits=[[949.0, 50.0, 1.0]],
    weights=[1.0],
    num_workers=4,
)
