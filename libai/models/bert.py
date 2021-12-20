from oneflow import nn
from libai.layers.transformer_layer import TransformerLayer
from libai.layers import VocabEmbedding
from libai.config import configurable


class Bert(nn.Module):
    @configurable
    def __init__(
        self, num_vocab, hidden_size, num_layers, blocks, add_pooler=True,
    ) -> None:
        super().__init__()

        self.embedding = VocabEmbedding(num_vocab, hidden_size)
        self.num_layers = num_layers
        self.add_pooler = add_pooler

        for i, block in enumerate(blocks):
            name = f"layer_{i}"
            self.add_module(name, block)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_vocab": cfg.num_embeddings,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "blocks": Bert.make_default_blocks(12),
            "add_pooler": cfg.add_pooler,
        }

    @staticmethod
    def make_default_blocks(num_layers, layer_class=None, **kwargs):
        if layer_class is None:
            layer_class = TransformerLayer

        layers = []
        for i in range(num_layers):
            kwargs["layer_idx"] = i
            layers.append(layer_class(**kwargs))
        return layers
