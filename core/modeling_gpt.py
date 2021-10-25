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
from core import distribute as dist
from core.utils import init_method_normal, scaled_init_method_normal
from core.module import ParallelEmbedding, ColumnParallelLinear, RowParallelLinear, ParallelMLP, ParallelLogits
from core.attention import SelfAttention


class GPTModel(flow.nn.Module):
    """GPT-2 language model. The output of the forward method is logits.
    
    Arguments:
        num_layers: number of layers.
        vocab_size: size of vocabulary.
        hidden_size: size of hidden state.
        num_attention_heads: number of attention heads.
        max_seq_length: maximum size of sequence, which is used for positional embedding.
        embedding_dropout_prob: dropout probability of embedding.
        attention_dropout_prob: dropout probability of attention weights.
        output_dropout_prob: dropout probability of output.
        layernorm_epsilon: epsilon used in layernorm.
        enable_amp: whether apply auto mixed precision (amp).
        checkpoint_activations: if `true`, checkpoint activations.
        use_scaled_init_for_output_weights: If `true`, use 1 / sqrt(2 * num_layers) scaling for the output weights.
        apply_query_key_layer_scaling: if `true`, scaling the attention score by layer index.
        bias_gelu_fusion: whether fuse add bias and gelu.
        bias_dropout_fusion: whether fuse add bias and dropout.
    """
    def __init__(self, num_layers, vocab_size, hidden_size, num_attention_heads, max_seq_length=1024, 
                 embedding_dropout_prob=0., attention_dropout_prob=0., output_dropout_prob=0.,
                 layernorm_epsilon=1e-5, enable_amp=False, checkpoint_activations=False,
                 use_scaled_init_for_output_weights=False, apply_query_key_layer_scaling=False,
                 bias_gelu_fusion=False, bias_dropout_fusion=False):
        super().__init__()
        init_method = init_method_normal(std=0.02)

        self.embedding = ParallelEmbedding(hidden_size, 
                                           vocab_size, 
                                           max_seq_length, 
                                           embedding_dropout_prob=embedding_dropout_prob,
                                           init_method=init_method,
                                           enable_amp=enable_amp)
        
        self.transformer = Transformer(num_layers, hidden_size, num_attention_heads, 
                                       attention_dropout_prob=attention_dropout_prob,
                                       output_dropout_prob=output_dropout_prob,
                                       layernorm_epsilon=layernorm_epsilon,
                                       checkpoint_activations=checkpoint_activations,
                                       use_scaled_init_for_output_weights=use_scaled_init_for_output_weights,
                                       apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                                       bias_gelu_fusion=bias_gelu_fusion,
                                       bias_dropout_fusion=bias_dropout_fusion)

        self.logits = ParallelLogits()

    def forward(self, token_ids, attention_mask, position_ids=None, layer_past=None, use_cache=False):
        embeddings = self.embedding(token_ids, position_ids=position_ids)
        
        transformer_output = self.transformer(embeddings, attention_mask, 
                                              layer_past=layer_past, use_cache=use_cache)

        if use_cache:
            transformer_output, presents = transformer_output
        
        output = self.logits(transformer_output, self.embedding.word_embeddings)

        if use_cache:
            output = [output, presents]
        return output


class Transformer(flow.nn.Module):
    """Transformer model for GPT-2. 

    It contains L (num_layers) blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followd by a final layer norm.
    
    Arguments:
        num_layers: number of layers.
        hidden_size: size of hidden state.
        num_attention_heads: number of attention heads.
        attention_dropout_prob: dropout probability of attention weights.
        output_dropout_prob: dropout probability of output.
        layernorm_epsilon: epsilon used in layernorm.
        init_method_std: standard deviation of the init method.
        checkpoint_activations: if `true`, checkpoint activations.
        use_scaled_init_for_output_weights: If `true`, use 1 / sqrt(2 * num_layers) scaling for the output weights.
        apply_query_key_layer_scaling: if `true`, scaling the attention score by layer index.
        bias_gelu_fusion: whether fuse add bias and gelu.
        bias_dropout_fusion: whether fuse add bias and dropout.
    """
    def __init__(self, num_layers, hidden_size, num_attention_heads, 
                 attention_dropout_prob=0., output_dropout_prob=0., 
                 layernorm_epsilon=1e-5, 
                 init_method_std=0.02,
                 checkpoint_activations=False,
                 use_scaled_init_for_output_weights=True,
                 apply_query_key_layer_scaling=False,
                 bias_gelu_fusion=False,
                 bias_dropout_fusion=False):
        super().__init__()
        self.num_layers = num_layers
        self.checkpoint_activations = checkpoint_activations

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method_normal(init_method_std, num_layers)

        def _build_layers():
            for i in range(num_layers):
                setattr(
                    self,
                    f"layer_{i}",
                    TransformerLayer(
                        i,
                        hidden_size,
                        num_attention_heads,
                        attention_dropout_prob=attention_dropout_prob,
                        output_dropout_prob=output_dropout_prob,
                        layernorm_epsilon=layernorm_epsilon,
                        init_method=init_method_normal(init_method_std),
                        output_layer_init_method=output_layer_init_method,
                        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                        bias_gelu_fusion=bias_gelu_fusion,
                        bias_dropout_fusion=bias_dropout_fusion,
                    ),
                )
                if self.checkpoint_activations:
                    setattr(self, f"layer_checkpoint_{i}", ActivationCheckpointing(i))

        self._build_layers()
        self.layernorm_f = LayerNorm(-1, hidden_size, eps=layernorm_epsilon)

    def _get_layer(self, layer_idx):
        layer = getattr(self, f"layer_{layer_idx}")
        checkpoint = None
        if self.checkpoint_activations:
            checkpoint = getattr(self, f"layer_checkpoint_{layer_idx}")
        return layer, checkpoint

    def forward(self, hidden_states, attention_mask, layer_past=None, use_cache=False):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # sbp: [S(0), B]
        
        if self.checkpoint_activations and not use_cache:
            for i in range(self.num_layers):
                layer, checkpoint = self._get_layer(i)                
                hidden_states = layer(checkpoint(hidden_states), attention_mask)     
        else:
            presents = []
            for i in range(self.num_layers):
                layer, _ = self._get_layer(i)
                past = None
                if layer_past is not None:
                    past = layer_past[i]
                hidden_states = layer(hidden_states, attention_mask, layer_past=past, use_cache=use_cache)
                if use_cache:
                    hidden_states, present = hidden_states
                    presents.append(present)

        output = self.layernorm_f(hidden_states)    
        if use_cache:
            output = [output, presents]
        return output


class TransformerLayer(flow.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [batch size, seq len, hidden size] and returns an
    output of the same size.

    The input and output's sbps are both [S(0), B].

    Arguments:
        layer_idx: the layer index, which determines the placement.
        hidden_size: size of hidden state.
        num_attention_heads: number of attention heads.
        attention_dropout_prob: dropout probability of attention weights.
        output_dropout_prob: dropout probability of output.
        init_method: method to initialize the input layer weights.
        output_layer_init_method: method to initialize the output layer weights. If None, use `init_method`.
        apply_query_key_layer_scaling: if `true`, scaling the attention score by layer index.
        bias_gelu_fusion: whether fuse add bias and gelu.
        bias_dropout_fusion: whether fuse add bias and dropout.
    """
    def __init__(self, layer_idx, hidden_size, num_attention_heads, 
                 attention_dropout_prob=0., output_dropout_prob=0.,
                 layernorm_epsilon=1e-5,
                 init_method=init.xavier_normal_,
                 output_layer_init_method=None,
                 apply_query_key_layer_scaling=False,
                 bias_gelu_fusion=False,
                 bias_dropout_fusion=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.layernorm_1 = LayerNorm(layer_idx, hidden_size, eps=layernorm_epsilon)

        self.attention = SelfAttention(layer_idx, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method, 
                                       output_layer_init_method=output_layer_init_method, 
                                       apply_query_key_layer_scaling=apply_query_key_layer_scaling)
        
        self.layernorm_2 = LayerNorm(layer_idx, hidden_size, eps=layernorm_epsilon)

        self.mlp = ParallelMLP(layer_idx, hidden_size, output_dropout_prob, init_method, 
                               output_layer_init_method=output_layer_init_method,
                               bias_gelu_fusion=bias_gelu_fusion,
                               bias_dropout_fusion=bias_dropout_fusion)


    def forward(self, hidden_states, attention_mask, layer_past=None, use_cache=False):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # sbp: [S(0), B]
        
        layernorm_output = self.layernorm_1(hidden_states)
        attention_output = self.attention(layernorm_output, attention_mask, layer_past, use_cache)
        
        if use_cache:
            attention_output, presents = attention_output
        layernorm_input = hidden_states + attention_output

        layernorm_output = self.layernorm_2(layernorm_input)
        mlp_output = self.mlp(layernorm_output)
        output = layernorm_input + mlp_output
        
        if use_cache:
            output = [output, presents]
        return output


class ActivationCheckpointing(flow.nn.Module):
    """Checkpoint for activations.
    """
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, x):
        x = x.to_consistent(placement=dist.get_layer_placement(self.layer_idx))
        return flow._C.identity(x)