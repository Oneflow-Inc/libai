from libai.config import LazyCall
from omegaconf import OmegaConf
from libai.data.build import build_nlp_test_loader, build_nlp_train_val_test_loader
from tests.data.datasets.demo_dataset import DemoNlpDataset

tokenizer = dict(
    tokenizer_name="BertTokenizer",
    tokenizer_cfg=dict(vocab_file="bert-vocab.txt", do_lower_case=True,),
    append_eod=False,
)

dataloader = OmegaConf.create()

dataloader.train=LazyCall(build_nlp_train_val_test_loader)(
    dataset=[
        LazyCall(DemoNlpDataset)(data_root="train1",),
        LazyCall(DemoNlpDataset)(data_root="train2",),
    ],
    splits=[[949.0, 50.0, 1.0], [900.0, 99.0, 1.0]],
    weights=[0.5, 0.5],
    batch_size=32,
    num_workers=4,
)

dataloader.test=[
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(DemoNlpDataset)(data_root="test1",), batch_size=32,
    ),
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(DemoNlpDataset)(data_root="test2",), batch_size=32,
    ),
]

