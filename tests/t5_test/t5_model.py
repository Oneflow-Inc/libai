from libai.layers.layer_norm import LayerNorm
from libai.layers.transformer_layer import TransformerLayer
from libai.layers.lm_logits import LMLogits
import oneflow as flow

from t5_model_layers.t5_model_embedding import T5Embedding

from libai.models.utils import init_method_normal, scaled_init_method_normal

class T5Model(flow.nn.Module):
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


if __name__ == '__main__':
    HIDDEN_SIZE=768
    VOCAB_SIZE=21248
    NUM_ATTENTION_HEADS=12
    FFN_HIDDEN_SIZE=3072
    ENCODER_SEQ_LENGTH=512
    DECODER_SEQ_LENGTH=128

    flow_t5 = T5Model(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        hidden_layers=12,
        num_attention_heads=NUM_ATTENTION_HEADS,
        intermediate_size=FFN_HIDDEN_SIZE,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=ENCODER_SEQ_LENGTH,
        apply_query_key_layer_scaling=False,
    )
    flow_t5.eval()

    from get_megatron_t5 import get_t5_model
    from utils import load_megatron_weight, get_sample, get_random_sample
    
    megatron_t5 = get_t5_model()
    load_megatron_weight(flow_t5, megatron_t5)

    
    FLOW_FM = None
    def flow_hook(module, inp, out):
        print('hook fuck')
        global FLOW_FM
        if FLOW_FM is None:
            FLOW_FM = out

    MEGA_FM = None
    def mega_hook(module, inp, out):
        global MEGA_FM
        if MEGA_FM is None:
            MEGA_FM = out

    tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = get_sample('flow')
    with flow.no_grad():
        flow_output = flow_t5(
            tokens_enc,
            tokens_dec,
            enc_mask,
            dec_mask,
            enc_dec_mask,
        )
    
    tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = get_sample('torch')
    import torch
    with torch.no_grad():
        megatron_output = megatron_t5(
            tokens_enc,
            tokens_dec,
            enc_mask,
            dec_mask,
            enc_dec_mask,
        )
    import numpy as np
    print('max diff: ', np.max(np.max(flow_output.numpy() - megatron_output.detach().cpu().numpy())))


    diff_results = []
    for i in range(100):
        flow_tensor, torch_tensor = get_random_sample(VOCAB_SIZE)
        tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = flow_tensor
        with flow.no_grad():
            flow_output = flow_t5(
                tokens_enc,
                tokens_dec,
                enc_mask,
                dec_mask,
                enc_dec_mask,
            )

        tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = torch_tensor
        with torch.no_grad():
            megatron_output = megatron_t5(
                tokens_enc,
                tokens_dec,
                enc_mask,
                dec_mask,
                enc_dec_mask,
            )
        print(f'{i}th max diff: ', np.max(np.max(flow_output.numpy() - megatron_output.detach().cpu().numpy())))
        diff_results.append(np.max(np.max(flow_output.numpy() - megatron_output.detach().cpu().numpy()))) 
    print('avg max diff: ', np.average(diff_results))