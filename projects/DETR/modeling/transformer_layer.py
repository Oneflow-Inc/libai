import oneflow as flow
import oneflow.nn as nn

from libai.layers.mlp import MLP
from libai.layers.transformer_layer import TransformerLayer

from .attention import DetrMultiheadAttention as MultiheadAttention


class DetrTransformerLayer(TransformerLayer):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        is_decoder=False,
        dropout_prob=0.0,
        normalize_before=False,
    ):

        super(DetrTransformerLayer, self).__init__(
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            is_decoder=is_decoder,
        )

        self.normalize_before = normalize_before

        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.self_attention = MultiheadAttention(
            self.hidden_size, self.num_attention_heads, attention_dropout_prob=dropout_prob
        )

        if self.is_decoder:
            self.cross_attention = MultiheadAttention(
                self.hidden_size, self.num_attention_heads, attention_dropout_prob=dropout_prob
            )
            self.dropout3 = nn.Dropout(dropout_prob)

        self.mlp = MLPLayer(
            self.hidden_size,
            self.ffn_hidden_size,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_padding_mask=None,
        memory=None,
        memory_mask=None,
        memory_key_padding_mask=None,
        position_embedding=None,
        query_position_embedding=None,
    ):
        """
        Args:
            hidden_states: shape is (batch_size, seq_length, hidden_size),
                sbp signature is (S(0), B).
            attention_mask: the combination of key padding mask and casual mask of hidden states
                with shape (batch_size, 1, seq_length, seq_length) and the sbp
                signature is (S(0), B),
        """

        if self.normalize_before:
            layernorm_hidden_states = self.input_layernorm(hidden_states)
            if self.is_decoder:
                query = key = self.with_pos_embed(layernorm_hidden_states, query_position_embedding)
            else:
                query = key = self.with_pos_embed(layernorm_hidden_states, position_embedding)
        else:
            if self.is_decoder:
                query = key = self.with_pos_embed(hidden_states, query_position_embedding)
            else:
                query = key = self.with_pos_embed(hidden_states, position_embedding)
        attention_output = self.self_attention(
            (query, key, hidden_states),
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        attention_output = self.dropout1(attention_output)

        hidden_states = hidden_states + attention_output

        if self.normalize_before:
            layernorm_output = self.post_attention_layernorm(hidden_states)
        else:
            layernorm_output = self.input_layernorm(hidden_states)

        if self.is_decoder:
            attention_output = self.cross_attention(
                (
                    self.with_pos_embed(layernorm_output, query_position_embedding),
                    self.with_pos_embed(memory, position_embedding),
                    memory,
                ),
                attention_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )
            attention_output = self.dropout3(attention_output)
            hidden_states = layernorm_output + attention_output
            layernorm_output = self.post_cross_attention_layernorm(hidden_states)

        mlp_output = self.mlp(layernorm_output)
        mlp_output = self.dropout2(mlp_output)

        if self.apply_residual_post_layernorm:
            residual = hidden_states
        else:
            residual = layernorm_output

        output = residual + mlp_output

        if not self.normalize_before:
            output = self.post_attention_layernorm(output)
        return output

    def with_pos_embed(self, tensor, pos):

        return tensor if pos is None else tensor + pos


class MLPLayer(MLP):
    def __init__(self, hidden_size, ffn_hidden_size):
        super(MLPLayer, self).__init__(hidden_size, ffn_hidden_size)

        self.activation_func = nn.ReLU()

    def forward(self, hidden_states):

        output = self.dense_4h_to_h(
            self.dropout(self.activation_func(self.dense_h_to_4h(hidden_states)))
        )

        return output
