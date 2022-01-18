# from libai.layers.layer_norm import LayerNorm
# from libai.layers.transformer_layer import TransformerLayer
# from libai.layers.lm_logits import LMLogits
<<<<<<< Updated upstream
import oneflow as flow
=======
>>>>>>> Stashed changes
from libai.config import configurable
from libai.layers import (
    LayerNorm,
    LMLogits,
    Embedding,
    VocabEmbedding,
    TransformerLayer,
    ParallelCrossEntropyLoss,
    lm_logits,
)
<<<<<<< Updated upstream
from libai.models.build import MODEL_ARCH_REGISTRY, GRAPH_REGISTRY
from libai.models.utils import init_method_normal, scaled_init_method_normal, GraphBase

from libai.utils import distributed as dist
=======
from libai.models.build import MODEL_ARCH_REGISTRY
import oneflow as flow

from libai.utils import distributed as dist
from libai.models.utils import init_method_normal, scaled_init_method_normal
>>>>>>> Stashed changes

class T5ExtendedAttnMask(flow.nn.Module):
    def forward(self, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # [b, 1, s]
        attention_mask_b1s = attention_mask.unsqueeze(1)
        # [b, s, 1]
        attention_mask_bs1 = attention_mask.unsqueeze(2)
        # [b, s, s]
        attention_mask_bss = attention_mask_b1s * attention_mask_bs1
        # [b, 1, s, s]
        extended_attention_mask = attention_mask_bss.unsqueeze(1)

        # Convert attention mask to binary.
        extended_attention_mask = flow.le(extended_attention_mask, 0.5)

        return extended_attention_mask


class T5Embedding(flow.nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method=flow.nn.init.xavier_normal_,
        num_tokentypes=0,
        fp16=False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_tokentypes = num_tokentypes

        self.word_embeddings = VocabEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            init_method=init_method,
            fp16=fp16,
        )
        self.position_embeddings = Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=hidden_size,
            init_method=init_method,
            fp16=fp16,
        )
        self.position_ids = flow.arange(
            max_sequence_length,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        ).unsqueeze(0)

        if self.num_tokentypes > 0:
            self.tokentype_embedding = Embedding(
                num_embeddings=self.num_tokentypes,
                embedding_dim=self.hidden_size,
                init_method=init_method,
                fp16=fp16,
            )
            self.tokentype_ids = flow.zeros(
                self.position_ids.size(),
                dtype=flow.long,
                sbp=self.position_ids.sbp,
                placement=self.position_ids.placement,
            )
        else:
            self.tokentype_embedding = None
            self.tokentype_ids = None

        self.embedding_dropout = flow.nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids, tokentype_ids=None, position_ids=None):
        # input_ids shape: [batch_size, seq_len, hidden_size]
        seq_length = input_ids.size()[1]
        word_embeddings = self.word_embeddings(input_ids)

        if position_ids is None:
            # Change position_ids sbp sign: [B, B] -> [S(0), B]
            position_ids = (
                self.position_ids[:, :seq_length]
                .expand_as(input_ids)
                .to_consistent(sbp=input_ids.sbp)
            )
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings

        if self.tokentype_embedding is not None:
            if tokentype_ids is None:
                tokentype_ids = (
                    self.tokentype_ids[:, :seq_length]
                    .expand_as(input_ids)
                    .to_consistent(sbp=input_ids.sbp)
                )
            embeddings = embeddings + self.tokentype_embedding(tokentype_ids)
        embeddings = self.embedding_dropout(embeddings)
        return embeddings
        


class T5Model(flow.nn.Module):

    @configurable
    def __init__(
        self,
        vocab_size,
        hidden_size,
        hidden_layers,
        num_attention_heads,
        intermediate_size,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        max_position_embeddings,
        num_tokentypes=0,
        initializer_range=0.02,
        layernorm_eps=1e-12,
        bias_gelu_fusion=False,
        bias_dropout_fusion=False,
        scale_mask_softmax_fusion=False,
        apply_query_key_layer_scaling=True,
        fp16=False
    ) -> None:
        super().__init__()
        init_method = init_method_normal(initializer_range)
        scaled_init_method = scaled_init_method_normal(initializer_range, hidden_layers)
        self.embedding = T5Embedding(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            max_sequence_length=max_position_embeddings,
            embedding_dropout_prob=hidden_dropout_prob,
            num_tokentypes=num_tokentypes,
            init_method=init_method,
            fp16=fp16,
        )
        self.extended_attn_mask = T5ExtendedAttnMask()

        encoder_layers = flow.nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    ffn_hidden_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    is_decoder=False,
                    attention_dropout_prob=attention_probs_dropout_prob,
                    output_dropout_prob=hidden_dropout_prob,
                    layernorm_epsilon=layernorm_eps,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    bias_gelu_fusion=bias_gelu_fusion,
                    bias_dropout_fusion=bias_dropout_fusion,
                    scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    layer_idx=i,
                )
                for i in range(hidden_layers)
            ]
        )

        encoder_final_layernorm = LayerNorm(
            (hidden_size, ), eps=layernorm_eps, layer_idx=-1
        )

        # for loading weight of Megatron
        self.encoder = flow.nn.Sequential()
        self.encoder.add_module('layers', encoder_layers)
        self.encoder.add_module('final_layernorm', encoder_final_layernorm)


        decoder_layers = flow.nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    ffn_hidden_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    is_decoder=True,
                    attention_dropout_prob=attention_probs_dropout_prob,
                    output_dropout_prob=hidden_dropout_prob,
                    layernorm_epsilon=layernorm_eps,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    bias_gelu_fusion=bias_gelu_fusion,
                    bias_dropout_fusion=bias_dropout_fusion,
                    scale_mask_softmax_fusion=scale_mask_softmax_fusion,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    layer_idx=i,
                )
                for i in range(hidden_layers)
            ]
        )

        decoder_final_layernorm = LayerNorm(
            (hidden_size, ), eps=layernorm_eps, layer_idx=-1
        )

        self.decoder = flow.nn.Sequential()
        self.decoder.add_module('layers', decoder_layers)
        self.decoder.add_module('final_layernorm', decoder_final_layernorm)

        self.lm_head = LMLogits(vocab_size, bias=True)

    @classmethod
    def from_config(cls, cfg):
        return {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "hidden_layers": cfg.hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "intermediate_size": cfg.intermediate_size,
            "hidden_dropout_prob": cfg.hidden_dropout_prob,
            "attention_probs_dropout_prob": cfg.attention_probs_dropout_prob,
            "max_position_embeddings": cfg.max_position_embeddings,
            "num_tokentypes": cfg.num_tokentypes,
            "initializer_range": cfg.initializer_range,
            "layernorm_eps": cfg.layernorm_eps,
            "bias_gelu_fusion": cfg.bias_gelu_fusion,
            "bias_dropout_fusion": cfg.bias_dropout_fusion,
            "scale_mask_softmax_fusion": cfg.scale_mask_softmax_fusion,
            "apply_query_key_layer_scaling": cfg.apply_query_key_layer_scaling,
            "fp16": cfg.fp16,
        }

    def forward(
        self, 
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
        tokentype_ids=None,
        # enc_hidden_states=None,
    ):
        # encoder_attn_mask = self.extended_attn_mask(encoder_attn_mask)
        # decoder_attn_mask = self.extended_attn_mask(decoder_attn_mask)
        # encoder_decoder_attn_mask = self.extended_attn_mask(encoder_decoder_attn_mask)

        enc_embedding_output = self.embedding(encoder_input_ids, tokentype_ids)
        enc_hidden_states = enc_embedding_output
        for layer in self.encoder.layers:
            enc_hidden_states = layer(enc_hidden_states, encoder_attn_mask)
        encoder_states = self.encoder.final_layernorm(enc_hidden_states)

        dec_embedding_output = self.embedding(decoder_input_ids, tokentype_ids)
        dec_hidden_states = dec_embedding_output
        for layer in self.decoder.layers:
            dec_hidden_states = layer(
                dec_hidden_states, 
                decoder_attn_mask, 
                encoder_states,
                encoder_decoder_attn_mask,
            )
        decoder_states = self.decoder.final_layernorm(dec_hidden_states)
        logits = self.lm_head(decoder_states, self.embedding.word_embeddings.weight)
        return logits


<<<<<<< Updated upstream
class T5Loss(flow.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_loss = ParallelCrossEntropyLoss()

    def forward(self, logits, lm_labels, loss_mask):
        lm_loss = self.lm_loss(logits, lm_labels)
        loss_mask = loss_mask.float()
        denominator = loss_mask.sum().to_consistent(
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast])
        )
        masked_lm_loss = flow.sum(lm_loss.view(-1) * loss_mask.view(-1)) / denominator
        return masked_lm_loss



=======
>>>>>>> Stashed changes
@MODEL_ARCH_REGISTRY.register()
class T5ForPreTraining(flow.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.t5_model = T5Model(cfg)
<<<<<<< Updated upstream
        self.loss_func = T5Loss()
=======
        self.loss_func = ParallelCrossEntropyLoss()
>>>>>>> Stashed changes
    
    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
<<<<<<< Updated upstream
=======
        tokentype_ids=None,
>>>>>>> Stashed changes
        lm_labels=None,
        loss_mask=None,
    ):
        logits = self.t5_model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attn_mask,
            decoder_attn_mask,
            encoder_decoder_attn_mask,
<<<<<<< Updated upstream
            # tokentype_ids=None,
            None,
        )

        lm_loss = self.loss_func(logits, lm_labels, loss_mask)
        return lm_loss

@GRAPH_REGISTRY.register()
class T5ForPretrainingGraph(GraphBase):
    def build(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
        lm_labels=None,
        loss_mask=None,
    ):

        # Forward pass through the model
        if self.is_train:
            losses = self.model(
                encoder_input_ids,
                decoder_input_ids,
                encoder_attn_mask,
                decoder_attn_mask,
                encoder_decoder_attn_mask,
                lm_labels,
                loss_mask,
            )
            losses.backward()
            return losses
=======
            tokentype_ids=tokentype_ids,
        )

        lm_loss = self.loss_func(logits, lm_labels)
        return lm_loss
>>>>>>> Stashed changes
