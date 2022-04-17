from distutils.command.config import config
from libai.config import configurable
import oneflow as flow
from . import get_model
from .losses import get_loss, CrossEntropyLoss_sbp
from .partial_fc import PartialFC


class Model(flow.nn.Module):

    @configurable
    def __init__(self, backbone, head, num_classes, sample_rate,
                 embedding_size) -> None:
        super().__init__()
        self.backbone = get_model(backbone, num_features=embedding_size)
        self.fc = PartialFC(embedding_size=embedding_size,
                            num_classes=num_classes,
                            sample_rate=1)
        self.head = get_loss(head)
        self.loss = CrossEntropyLoss_sbp()

    @classmethod
    def from_config(cls, cfg):
        return {
            "backbone": cfg.backbone,
            "head": cfg.head,
            "num_classes": cfg.num_classes,
            "sample_rate": cfg.sample_rate,
            "embedding_size": cfg.embedding_size,
        }

    def forward(self, images, labels):
        logits = self.backbone(images)
        logits, map_labels = self.fc(logits, labels)
        # import ipdb
        # ipdb.set_trace()
        if self.training:
            logits = self.head(logits, labels)
            loss = self.loss(logits, labels)
            return {"loss": loss}
        else:
            return {"prediction_scores": logits}