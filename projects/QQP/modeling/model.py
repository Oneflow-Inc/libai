from libai.models.utils import init_method_normal, GraphBase
from libai.models.bert_model import BertModel
from libai.layers import Linear
from oneflow import nn
import oneflow as flow
from libai.utils import distributed as dist
from .load_megatron_weight import load_megatron_bert
import logging


logger = logging.getLogger("libai."+__name__)

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, classification_logits, label):
        loss = nn.CrossEntropyLoss()(classification_logits, label)
        # NOTE: Change loss sbp sign [P, P] -> [P, B] to add with sop loss
        # whose sbp sign: [P, B]
        loss = loss.to_consistent(
            sbp=dist.get_nd_sbp([flow.sbp.partial_sum, flow.sbp.broadcast])
        )
        return loss

class Classification(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.language_model = BertModel(cfg)
        if cfg.pretrain_megatron_weight is not None:
            logger.info(f"loading pretraining: {cfg.pretrain_megatron_weight}")
            load_megatron_bert(self.language_model, cfg.pretrain_megatron_weight)
            logger.info(f"load succeed")

        init_method = init_method_normal(cfg.initializer_range)
        self.classification_dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.classification_head = Linear(
            cfg.hidden_size,
            self.num_classes,
            bias=True,
            parallel="row",
            init_method=init_method,
            layer_idx=-1,
        )
        self.loss_func = ClassificationLoss()
    
    def forward(self, model_input, attention_mask, tokentype_ids=None, label=None):
        
        encoder_output, pooled_output = self.language_model(model_input, attention_mask, tokentype_ids)
        classification_output = self.classification_dropout(pooled_output)
        classification_logits = self.classification_head(classification_output)
        
        # reshape
        classification_logits = classification_logits.view(-1, self.num_classes)
    
        if label is not None:
            loss =  self.loss_func(classification_logits, label)
            return loss
        
        return classification_logits
    
    
class ClassificationGraph(GraphBase):
    def build(self, tokens, padding_mask, tokentype_ids, label=None):
        if not self.is_train:
            return self.model(tokens, padding_mask, tokentype_ids)
        else:
            losses = self.model(
                tokens, padding_mask, tokentype_ids, label
            )

            losses.backward()
            return losses