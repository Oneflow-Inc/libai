from platform import java_ver
import oneflow as flow
import torch
import numpy as np

from t5_model_layers.t5_model_embedding import T5Embedding
from t5_model_layers.t5_model_encoder import T5EncoderLayer
from get_megatron_t5 import get_t5_model
from megatron.model.t5_model import T5Model, t5_position_ids
from torch.nn.modules.dropout import FeatureAlphaDropout 
from utils import convert_and_copy_tensor, numpy_to_flow, load_megatron_embedding_weight, get_sample

HIDDEN_SIZE=768
VOCAB_SIZE=21248
NUM_ATTENTION_HEADS=12
FFN_HIDDEN_SIZE=3072
ENCODER_SEQ_LENGTH=512
DECODER_SEQ_LENGTH=128
""" Megatron t5 config

    '--num-layers', '12',
    '--hidden-size', '768', 
    '--num-attention-heads', '12', 
    '--kv-channels', '64', 
    '--ffn-hidden-size', '3072', 
    '--encoder-seq-length', '512', 
    '--decoder-seq-length', '128', 
    '--micro-batch-size', '16', 
    '--global-batch-size', '16', 
    '--max-position-embeddings', '512', 
    '--train-iters', '1000000', 
    '--lr-decay-iters', '1000000', 
    '--save', '/home/wang/workspace/Megatron-LM/examples', 
    '--load', '/home/wang/workspace/Megatron-LM/examples', 
    '--data-path', '/home/wang/workspace/Megatron-LM/examples', 
    '--vocab-file', '/home/wang/data/t5/dataset/bert-base-chinese-vocab.txt', 
    '--data-impl', 'mmap', 
    '--split', '949,50,1', 
    '--lr', '0.0001', 
    '--min-lr', '0.00001', 
    '--lr-decay-style', 'linear', 
    '--lr-warmup-fraction', '.01', 
    '--weight-decay', '1e-2', 
    '--clip-grad', '1.0', 
    '--log-interval', '100', 
    '--save-interval', '10000', 
    '--eval-interval', '1000', 
    '--eval-iters', '10', 
    # '--fp16', 
    '--vocab-extra-ids', '100'
"""

if __name__ == '__main__':
    FEATURE_MAP = None
    model = get_t5_model()
    tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = get_sample('torch')

    def register_forward_hook(module, inp, out):
        global FEATURE_MAP
        if FEATURE_MAP is None:
            # print(module, out)
            FEATURE_MAP = out

    # embeddings
    embedding_module = model.language_model.embedding

    # embeddings hook
    # hook = model.language_model.embedding.register_forward_hook(register_forward_hook)
    tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = get_sample('torch')
    # with torch.no_grad():
    #     model(tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask)
    # hook.remove()

    # embeddings forward
    with torch.no_grad():
        position_ids = t5_position_ids(tokens_enc)
        embedding_module_output = embedding_module(tokens_enc, position_ids)
    

    # encoder preprocessing
    embedding_module_output.transpose_(0, 1)
    enc_mask.unsqueeze_(1)
    hidden_state = embedding_module_output

    # encoder forward
    for i in range(12):
        FEATURE_MAP = None
        encoder_layer_module = model.language_model.encoder.layers[i]

        # encoder hook
        hook = model.language_model.encoder.layers[i].register_forward_hook(register_forward_hook)
        tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = get_sample('torch')
        with torch.no_grad():
            model(tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask)
        hook.remove()

        with torch.no_grad():
            hidden_state = encoder_layer_module(hidden_state, enc_mask)
            # print(f'layer{i} hidden state: ', hidden_state)
            # print(f'layer{i} hook: ', FEATURE_MAP)
            print(f'layer{i} diff: ', np.max(np.abs((FEATURE_MAP - hidden_state).cpu().numpy())))
    

    # encoder final layernorm
    encoder_final_layermorm_module = model.language_model.encoder.final_layernorm
    FEATURE_MAP = None

    with torch.no_grad():
        hidden_state = encoder_final_layermorm_module(hidden_state)
        hook = model.language_model.encoder.final_layernorm.register_forward_hook(register_forward_hook)
        tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = get_sample('torch')
        with torch.no_grad():
            model(tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask)
        hook.remove()
        hidden_state.transpose_(0, 1)

    print(f'final_layernorm diff: ', np.max(np.abs((FEATURE_MAP - hidden_state).cpu().numpy())))

    


