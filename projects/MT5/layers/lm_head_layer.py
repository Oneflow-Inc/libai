from oneflow import nn

from libai.layers import Linear, LMLogits


class LMHead(nn.Module):
    def __init__(self, model_type, hidden_size, vocab_size, hidden_layers):
        super().__init__()
        if model_type == "mt5":
            self.lm_head = Linear(
                hidden_size, vocab_size, bias=False, layer_idx=2 * hidden_layers - 1
            )
        else:
            self.lm_head = LMLogits(vocab_size, bias=True)

    def forward(self, decoder_states, embed_weight=None):
        if isinstance(self.lm_head, Linear):
            logits = self.lm_head(decoder_states)
        else:
            logits = self.lm_head(decoder_states, embed_weight)
        return logits
