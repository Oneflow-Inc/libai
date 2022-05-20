import logging

import oneflow as flow
from oneflow import nn

from libai.layers import Linear
from libai.models.bert_model import BertModel
from libai.models.utils import init_method_normal
from libai.utils import distributed as dist
import pdb

logger = logging.getLogger("libai." + __name__)


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, classification_logits, label):
        loss = nn.CrossEntropyLoss()(classification_logits, label)
        # NOTE: Change loss sbp sign [P, P] -> [P, B] to add with sop loss
        # whose sbp sign: [P, B]
        loss = loss.to_global(sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast]))
        return loss


class ModelForSequenceClassification(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.model = BertModel(cfg)
        if cfg.pretrain_megatron_weight is not None:
            from .load_megatron_weight import load_megatron_bert

            logger.info(f"loading pretraining: {cfg.pretrain_megatron_weight}")
            load_megatron_bert(self.model, cfg.pretrain_megatron_weight)
            logger.info("load succeed")

        init_method = init_method_normal(cfg.initializer_range)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

        self.classifier = Linear(
            cfg.hidden_size,
            self.num_classes,
            bias=True,
            parallel="row",
            init_method=init_method,
            layer_idx=-1,
        )
        self.loss_fct = ClassificationLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):

        outputs= self.model(input_ids, attention_mask, token_type_ids)
        
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.training and labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            loss_dict = {"loss": loss}
            return loss_dict

        return {"prediction_scores": logits}



