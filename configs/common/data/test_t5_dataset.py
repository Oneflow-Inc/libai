from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.data.build import build_nlp_train_loader
from libai.data.datasets.test_t5_dataset import T5Dataset
from libai.data.data_utils import get_indexed_dataset


dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(T5Dataset)(vocab_size=25000, num_samples=1000, seq_len=512),
    ],
    num_workers=4,
)
