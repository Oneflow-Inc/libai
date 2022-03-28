import oneflow as flow
import oneflow.nn as nn

from typing import Any

from libai.config.config import configurable
from libai.layers import Linear
from libai.utils import distributed as dist

from .build import MODEL_ARCH_REGISTRY


@MODEL_ARCH_REGISTRY.register()
class AlexNet(nn.Module):
    @configurable
    def __init__(self, num_classes: int = 1000, loss_func=None,) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2).to_global(
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2).to_global(
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1).to_global(
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1).to_global(
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1).to_global(
                placement=dist.get_layer_placement(0),
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            Linear(256 * 6 * 6, 4096, layer_idx=-1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Linear(4096, 4096, layer_idx=-1),
            nn.ReLU(inplace=True),
            Linear(4096, num_classes, layer_idx=-1),
        )

        self.loss_func = nn.CrossEntropyLoss() if loss_func is None else loss_func
    
    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes": cfg.num_classes,
        }

    def forward(self, images, labels=None) -> flow.Tensor:
        x = self.features(images)
        x = x.to_global(placement=dist.get_layer_placement(-1))
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.classifier(x)

        if labels is not None and self.training:
            losses = self.loss_func(x, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": x}
