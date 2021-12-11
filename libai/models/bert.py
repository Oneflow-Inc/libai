from oneflow import nn
from libai.layers.transformer_layer import TransformerLayer


class Bert(nn.Module):
    def __init__(self, embedding, blocks, add_pooler=True,) -> None:
        super().__init__()

        self.embedding = embedding
        self.add_pooler = add_pooler

        for i, block in enumerate(blocks):
            name = f"layer_{i}"
            self.add_module(name, block)

    @staticmethod
    def make_default_blocks(num_layers, layer_class=None, **kwargs):
        if layer_class is None:
            layer_class = TransformerLayer

        layers = []
        for i in range(num_layers):
            kwargs["layer_idx"] = i
            layers.append(layer_class(**kwargs))
        return layers
