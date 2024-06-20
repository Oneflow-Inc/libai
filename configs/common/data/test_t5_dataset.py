from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.data.build import build_nlp_train_loader
from libai.data.datasets.test_t5_dataset import T5Dataset
from libai.data.data_utils import get_indexed_dataset


dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(T5Dataset)(vocab_size=8, num_samples=1024, enc_seq_len=8, dec_seq_len=8),
    ],
    num_workers=4,
)
