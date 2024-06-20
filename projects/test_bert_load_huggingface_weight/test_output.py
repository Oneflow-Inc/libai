import unittest

import numpy as np
import oneflow as flow
import torch
import transformers
from load_huggingface_weight import load_huggingface_bert

import libai
from libai.config import LazyCall
from libai.models import build_model
from libai.utils import distributed as dist


class Test_BertModel_Use_Huggingface_Weight(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.input_ids = [[101, 1962, 2110, 739, 999, 1, 2, 3, 4, 102]]
        self.mask = [[1] * len(self.input_ids)]
        # libai's config
        self.cfg = dict(
            vocab_size=21128,
            hidden_size=768,
            hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            num_tokentypes=2,
            add_pooling_layer=True,
            initializer_range=0.02,
            layernorm_eps=1e-12,
            bias_gelu_fusion=False,
            bias_dropout_fusion=False,
            scale_mask_softmax_fusion=False,
            apply_query_key_layer_scaling=False,
            add_binary_head=True,
            amp_enabled=False,
            apply_residual_post_layernorm=True,
        )
        self.bert_libai = build_model(LazyCall(libai.models.BertModel)(cfg=self.cfg))
        load_huggingface_bert(
            self.bert_libai,
            "./bert-base-chinese/pytorch_model.bin",
            self.cfg["hidden_size"],
            self.cfg["num_attention_heads"],
        )
        self.input_ids_of = flow.tensor(
            self.input_ids,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),
        )
        self.mask_of = flow.tensor(
            self.mask,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
            placement=flow.placement("cuda" if flow.cuda.is_available() else "cpu", [0]),
        )

        # huggingface's config
        self.bert_huggingface = transformers.BertModel.from_pretrained("./bert-base-chinese")
        self.input_ids_pt = torch.tensor(self.input_ids)
        self.mask_pt = torch.tensor(self.mask)

    def test_output(self):
        # libai's bert
        self.bert_libai.eval()
        last_hidden_state_of = self.bert_libai(self.input_ids_of, self.mask_of)[0]

        # huggingface's Bert
        self.bert_huggingface.eval()
        last_hidden_state_pt = self.bert_huggingface(
            self.input_ids_pt, self.mask_pt
        ).last_hidden_state

        res1 = last_hidden_state_of.detach().numpy().sum()
        res2 = last_hidden_state_pt.detach().numpy().sum()

        self.assertTrue(np.around(res1, 4) == np.around(res2, 4))


if __name__ == "__main__":
    unittest.main()
