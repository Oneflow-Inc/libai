import oneflow as flow
from libai.layers import TransformerLayer


class T5EncoderLayer(flow.nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        

    ) -> None:
        super().__init__()