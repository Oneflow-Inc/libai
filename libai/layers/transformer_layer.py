# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import oneflow as flow
import oneflow.nn.init as init

from libai.utils import distributed as dist

from .attention import MultiheadAttention
from .layer_nrom import LayerNorm
from .mlp import MLP


class TransformerLayer(flow.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [seq_length, bsz, hidden size] and returns an
    output of the same size.
    The input and output has same sbp sign, (S(1), B).

    Arguments:
        hidden_size: size of hidden state.
        ffn_hidden_size: size of feed forword neural network.
        num_attention_heads: number of attention heads.
        is_decoder: used to specify whether this is transformer encoder layer or transformer decoder layer. Default: ``False``.
        attention_dropout_prob: dropout probability of attention weights.
        output_dropout_prob: dropout probability of output.
        layernorm_epsilon: epsilon used in layernorm layer. Default: `1e-5`.
        init_method: method to initialize the input layer weights.
        output_layer_init_method: method to initialize the output layer weights. If None, use `init_method`.
        bias_gelu_fusion: whether fuse add bias and gelu. Default: ``False``.
        bias_dropout_fusion: whether fuse add bias and dropout. Default: ``False``.
        scale_mask_softmax_fusion: whether to fuse scale, mask and softmax. Default: ``False``.
        apply_query_key_layer_scaling: if `true`, scaling the attention score by layer index. Default: ``False``.
        layer_idx: the layer index, which determines the placement.
    """
    def __init__(self, args, is_decoder=False, init_method=init.xavier_normal_, output_layer_init_method=None, *, layer_idx=0):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.layer_idx = layer_idx
        self.is_decoder = is_decoder

        self.init_method = init_method
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        self.input_layernorm = LayerNorm(args.hidden_size, eps=args.layernorm_epsilon, layer_idx=self.layer_idx)

        self.self_attention = self.build_self_attention(args)
        self.post_attention_layernorm = LayerNorm(args.hidden_size, eps=args.layernorm_epsilon, layer_idx=self.layer_idx)
        
        if self.is_decoder:
            self.cross_attention = self.build_cross_attention(args)
            self.post_cross_attention_layernorm = LayerNorm(args.hidden_size, eps=args.layernorm_epsilon, layer_idx=self.layer_idx)

        self.mlp = MLP(args.hidden_size, args.ffn_hidden_size, args.output_dropout_prob, self.init_method, 
                       output_layer_init_method=self.output_layer_init_method,
                       bias_gelu_fusion=args.bias_gelu_fusion,
                       bias_dropout_fusion=args.bias_dropout_fusion, 
                       layer_idx=self.layer_idx)


    def forward(self, hidden_states, attention_mask, 
                encoder_states=None, encoder_attention_mask=None, 
                past_key_value=None, use_cache=False):
        """ hidden_states: [seq_length, bsz, hidden_size], (S(1), B),
            attention_mask: [bsz, 1, seq_length, seq_length], (S(0), B), the combination of key padding mask and casual mask of hidden states.
            encoder_states: [seq_length, bsz, hidden_size], (S(1), B), encoder output, this will be used in cross attention.
            encoder_attention_mask: [bsz, 1, seq_length, seq_length], (S(1), B) key padding mask of encoder states.
            past_key_value: tuple of key and value, each shape is [src_len, bsz, num_heads, head_size]. For decoder layer,
                            the past_key_value contains the states both from self attention and cross attention.
            use_cache: it will be set to `True`, when the model is in the inference phase and used for incremental decoding.
        """
        # hidden_states shape: (seq_length, batch_size, hidden_size)
        attention_mask = attention_mask.to_consistent(placement=dist.get_layer_placement(self.layer_idx))
        
        if past_key_value is not None:
            if self.is_decoder:
                assert len(past_key_value) == 4 
                self_attn_past_key_value = past_key_value[:2]
                cross_attn_past_key_value = past_key_value[2:]
            else:
                self_attn_past_key_value = past_key_value
                cross_attn_past_key_value = None
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.self_attention(layernorm_output, 
                                               attention_mask=attention_mask, 
                                               past_key_value=self_attn_past_key_value, 
                                               use_cache=use_cache)
        
        if use_cache:
            attention_output, presents = attention_output
        hidden_states = hidden_states + attention_output

        layernorm_output = self.post_attention_layernorm(hidden_states)

        if self.is_decoder:
            attention_output = self.cross_attention(layernorm_output, encoder_states, 
                                                    attention_mask=encoder_attention_mask,
                                                    past_key_value=cross_attn_past_key_value, 
                                                    use_cache=use_cache)

            if use_cache:
                attention_output, decoder_presents = attention_output
                presents += decoder_presents
            
            hidden_states = hidden_states + attention_output
            layernorm_output = self.post_cross_attention_layernorm(hidden_states)

        mlp_output = self.mlp(layernorm_output)
        output = hidden_states + mlp_output
        
        if use_cache:
            output = (output, presents)
        return output
    
    def build_self_attention(self, args):
        return MultiheadAttention(args.hidden_size, args.num_attention_heads,
                                  is_cross_attention=False,
                                  attention_dropout_prob=args.attention_dropout_prob, 
                                  output_dropout_prob=args.output_dropout_prob, 
                                  init_method=self.init_method, 
                                  output_layer_init_method=self.output_layer_init_method, 
                                  bias_dropout_fusion=args.bias_dropout_fusion,
                                  scale_mask_softmax_fusion=args.scale_mask_softmax_fusion,
                                  apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
                                  layer_idx=self.layer_idx)

    def build_cross_attention(self, args):
        return MultiheadAttention(args.hidden_size, args.num_attention_heads,
                                  is_cross_attention=True,
                                  attention_dropout_prob=args.attention_dropout_prob, 
                                  output_dropout_prob=args.output_dropout_prob, 
                                  init_method=self.init_method, 
                                  output_layer_init_method=self.output_layer_init_method, 
                                  bias_dropout_fusion=args.bias_dropout_fusion,
                                  scale_mask_softmax_fusion=args.scale_mask_softmax_fusion,
                                  apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
                                  layer_idx=self.layer_idx)
