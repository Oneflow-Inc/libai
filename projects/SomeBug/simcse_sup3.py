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

            
            # bug3----------------------------------
            y_true = flow.arange(out.size(0), sbp=out.sbp, placement=out.placement, dtype=flow.long)  # arange引发的报错
            y_true = (y_true - y_true % 2 * 2) + 1
            

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