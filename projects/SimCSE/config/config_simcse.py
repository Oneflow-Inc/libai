from omegaconf import OmegaConf

from configs.common.data.bert_dataset import tokenization
from configs.common.models.bert import cfg as simcse_cfg
from configs.common.optim import optim
from configs.common.train import train
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from projects.SimCSE.dataset.dataset import TrainDataset, TestDataset, PadBatchData
from projects.SimCSE.modeling.simcse_unsup import SimcseModel
from libai.tokenizer import BertTokenizer


tokenization.tokenizer = LazyCall(BertTokenizer){
    vocab_file = "/home/xiezipeng/libai/projects/SimCSE/dataset/vocab.txt"
}

data_loader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(TrainDataset)(
            name="SimCse_TRAIN",
            datapaths=[
                "/home/chengpeng/train.tsv",
            ],
            max_seq_length=512,
        ),
    ],
    num_workers=4,
)