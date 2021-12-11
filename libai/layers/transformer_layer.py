import oneflow as flow
from oneflow import nn


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size=768, layer_idx=0) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
