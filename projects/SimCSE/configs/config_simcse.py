# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import OmegaConf

from configs.common.data.bert_dataset import tokenization
from configs.common.models.bert import cfg as simcse_cfg
from configs.common.models.graph import graph
from configs.common.optim import optim
from configs.common.train import train
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from projects.SimCSE.dataset.dataset import TrainDataset, TestDataset
from projects.SimCSE.modeling.simcse_unsup import SimcseModel
from libai.tokenizer import BertTokenizer


tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file = "/home/xiezipeng/libai/projects/SimCSE/dataset/vocab.txt"
)

dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(TrainDataset)(
            name="wiki",
            path="/home/xiezipeng/libai/projects/SimCSE/dataset/wiki1m_for_simcse.txt",
            tokenizer=LazyCall(BertTokenizer)(vocab_file = "/home/xiezipeng/libai/projects/SimCSE/dataset/vocab.txt")
        )
    ],

)

dataloader.test = [
    LazyCall(build_nlp_test_loader)(
        dataset=LazyCall(TestDataset)(
            name='sts',
            path='/home/xiezipeng/libai/projects/SimCSE/dataset/sts_test.txt',
            tokenizer=LazyCall(BertTokenizer)(vocab_file = "/home/xiezipeng/libai/projects/SimCSE/dataset/vocab.txt")
        ),
    ),
]

# dataloader.val = [
#     LazyCall(build_nlp_test_loader)(
#         dataset=LazyCall(TestDataset)(
#             name='sts',
#             path='/home/xiezipeng/libai/projects/SimCSE/dataset/sts_dev.txt',
#             tokenizer=LazyCall(BertTokenizer)(vocab_file = "/home/xiezipeng/libai/projects/SimCSE/dataset/vocab.txt")
#         )
#     )
# ]

simcse_cfg["intermediate_size"]=3072
simcse_cfg["hidden_layers"]=12
simcse_cfg["layernorm_eps"]=1e-12
simcse_cfg["pretrained_model_weight"]=None
simcse_cfg["pooler_type"]='cls'
simcse_cfg["temp"] = 0.05

model=LazyCall(SimcseModel)(cfg=simcse_cfg)

train.update(
    dict(
        output_dir="/home/xiezipeng/libai/projects/SimCSE/dataset",
        train_micro_batch_size=10,
        test_micro_batch_size=10,
        train_epoch=1,
        train_iter=15625,
        eval_period=50,
        dist=dict(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),      
    )
)

