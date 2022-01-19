from libai.models import T5Model
from libai.utils.checkpoint import Checkpointer
import oneflow as flow

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
        global FLOW_FM
        if FLOW_FM is None:
            FLOW_FM = out
    
    flow_t5.encoder.layers[0].self_attention.dense.register_forward_hook(flow_hook)

    MEGA_FM = None
    def mega_hook(module, inp, out):
        global MEGA_FM
        if MEGA_FM is None:
            MEGA_FM = out
    
    megatron_t5.language_model.encoder.layers[0].self_attention.dense.register_forward_hook(mega_hook)

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