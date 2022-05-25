from turtle import forward

import oneflow.nn as nn
import oneflow as flow

from libai.utils import distributed as dist
from libai.layers.transformer_layer import TransformerLayer
from libai.layers.mlp import MLP

from .attention import DetrMultiheadAttention as MultiheadAttention


class DetrTransformerLayer(TransformerLayer):
    
    def __init__(
        self, 
        hidden_size, 
        ffn_hidden_size, 
        num_attention_heads, 
        is_decoder=False,
        dropout_prob=0.0,
        apply_residual_post_layernorm=False
    ):
        
        super(DetrTransformerLayer, self).__init__(
            hidden_size, 
            ffn_hidden_size, 
            num_attention_heads, 
            is_decoder=is_decoder,
            apply_residual_post_layernorm=apply_residual_post_layernorm
        )    

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_attention_heads = num_attention_heads
        self.is_decoder = is_decoder
        self.apply_residual_post_layernorm = apply_residual_post_layernorm
        
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.self_attention = MultiheadAttention(
            self.hidden_size,
            self.num_attention_heads,
            attention_dropout_prob=dropout_prob
        )
        
        if self.is_decoder:
            self.cross_attention = MultiheadAttention(
                self.hidden_size,
                self.num_attention_heads,
                attention_dropout_prob=dropout_prob
            )
            self.dropout3 = nn.Dropout(dropout_prob)
            # self.post_cross_attention_layernorm = nn.LayerNorm(self.hidden_size)
            
        self.mlp = MLPLayer(
            self.hidden_size,
            self.ffn_hidden_size,
        )
            
        # self.input_layernorm = nn.LayerNorm(self.hidden_size)
        # self.post_attention_layernorm = nn.LayerNorm(self.hidden_size)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_padding_mask = None,
        memory=None,
        memory_mask=None,
        memory_key_padding_mask=None,
        position_embedding = None,
        query_position_embedding = None,
    ):
        """
        Args:
            hidden_states: shape is (batch_size, seq_length, hidden_size),
                sbp signature is (S(0), B).
            attention_mask: the combination of key padding mask and casual mask of hidden states
                with shape (batch_size, 1, seq_length, seq_length) and the sbp
                signature is (S(0), B),
            encoder_states: encoder output with shape (batch_size, seq_length, hidden_size)
                and the sbp signature is (S(0), B), which will be used in cross attention.
            encoder_attention_mask: key padding mask of encoder states with shape
                (batch_size, 1, seq_length, seq_length) and the sbp signature is (S(0), B).
            past_key_value: tuple of key and value, each shape is
                (seq_length, bsz, num_heads, head_size), For decoder layer,
                the past_key_value contains the states both from self attention
                and cross attention.
            use_cache: it will be set to `True` when the model is in the inference phase and
                used for incremental decoding.
        """
        # # Change placement for pipeline parallelsim
        # hidden_states = hidden_states.to_global(placement=dist.get_layer_placement(self.layer_idx))
        # position_embedding = position_embedding.to_global(placement=dist.get_layer_placement(self.layer_idx))
        
        # # hidden_states shape: (batch_size, seq_length, hidden_size)
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to_global(
        #         placement=dist.get_layer_placement(self.layer_idx)
        #     )
        # if key_padding_mask is not None:
        #     key_padding_mask = key_padding_mask.to_global(
        #         placement=dist.get_layer_placement(self.layer_idx)
        #     )
        if self.is_decoder:
            query = key = self.with_pos_embed(hidden_states, query_position_embedding)
        else:
            query = key = self.with_pos_embed(hidden_states, position_embedding)
        attention_output = self.self_attention(
            (query, key, hidden_states),
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask
        )
        attention_output = self.dropout1(attention_output)
        
        if self.apply_residual_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        hidden_states = residual + attention_output

        layernorm_output = self.input_layernorm(hidden_states)
        
        if self.is_decoder:
            attention_output = self.cross_attention(
                (self.with_pos_embed(layernorm_output, query_position_embedding),
                 self.with_pos_embed(memory, position_embedding), 
                 memory),
                attention_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
            attention_output = self.dropout3(attention_output)
            if self.apply_residual_post_layernorm:
                residual = hidden_states
            else:
                residual = layernorm_output
            hidden_states = residual + attention_output
            layernorm_output = self.post_cross_attention_layernorm(hidden_states)

        mlp_output = self.mlp(layernorm_output)
        mlp_output = self.dropout2(mlp_output)

        if self.apply_residual_post_layernorm:
            residual = hidden_states
        else:
            residual = layernorm_output

        output = residual + mlp_output
        
        output = self.post_attention_layernorm(output)

        return output
    
    def with_pos_embed(self, tensor, pos):
        
        return tensor if pos is None else tensor + pos
    
    
    
class MLPLayer(MLP):
    
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size):
        super(MLPLayer, self).__init__(hidden_size, ffn_hidden_size)
        
        self.activation_func = nn.ReLU()
    
    def forward(self, hidden_states):
        
        output = self.dense_4h_to_h(self.dropout(self.activation_func(self.dense_h_to_4h(hidden_states))))
        
        return output
        