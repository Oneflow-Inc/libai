import numpy as np
import oneflow as flow
from oneflow import nn
import libai
from libai.utils import distributed as dist


def cosine_similarity(x, y, dim=-1):
    return (
        flow.sum(x * y, dim=dim)
        / (flow.linalg.norm(x, dim=dim) * flow.linalg.norm(y, dim=dim))
    )


class Simcse(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = libai.models.BertModel(cfg)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if self.training:
            bs = input_ids.size(0)
            input_ids = input_ids.view(bs*3, -1)
            attention_mask = attention_mask.view(bs*3, -1)
            out = self.bert(input_ids, attention_mask)
            out = out[0][:, 0]
            sim = cosine_similarity(out.unsqueeze(1), out.unsqueeze(0))

            
            # bug2----------------------------------
            y_true = np.arange(out.size(0))
            use_row = np.where((y_true + 1) % 3 != 0)[0]
            use_row = flow.tensor(use_row, sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), placement=out.placement)
            y_true = (use_row - use_row % 3 * 2) + 1
            sim = sim - flow.eye(out.size(0), sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), placement=out.placement) * 1e12
            sim = flow.index_select(sim, 0, use_row)  # index_select引起报错
            sim = sim / 0.05
            y_true = flow.tensor(y_true, dtype=flow.long, sbp=out.sbp, placement=out.placement)
            

            loss = nn.CrossEntropyLoss()(sim, y_true)
            loss = loss.to_global(sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast]))
            return {"loss": loss}

        else:
            bs = input_ids.size(0)
            input_ids = input_ids.view(bs*2, -1)
            attention_mask = attention_mask.view(bs*2, -1)
            out = self.bert(input_ids, attention_mask)
            out = out[0][:, 0]
            out = out.view(bs, 2, -1)
            sent1 = out[:, 0]
            sent2 = out[:, 1] 
            sim = cosine_similarity(sent1, sent2)
            sim = sim.to_global(sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]))
            return {"sim": sim}