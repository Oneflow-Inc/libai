import oneflow as flow
import torch
import numpy as np

from t5_model_layers.t5_model_embedding import T5Embedding
from t5_model_layers.t5_model_encoder import T5EncoderLayer
from get_megatron_t5 import get_t5_model
from megatron.model.t5_model import T5Model, t5_position_ids 
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

def get_flow_t5_embedding():
    return T5Embedding(HIDDEN_SIZE, VOCAB_SIZE, ENCODER_SEQ_LENGTH, 0.1).eval()

def get_flow_t5_encoderlayer():
    return T5EncoderLayer(HIDDEN_SIZE, FFN_HIDDEN_SIZE, NUM_ATTENTION_HEADS, 0.1, 0, )


def get_megatron_t5_embedding():
    model = get_t5_model()
    model = model.language_model.embedding
    return model

def get_megatron_t5_encoder_layer():
    model = get_t5_model()
    model = model.language_model.encoder.layers[0]
    return model


def get_random_token_input():
    token = np.random.randint(0, VOCAB_SIZE, size=(1, ENCODER_SEQ_LENGTH), dtype=np.int64)
    flow_token = numpy_to_flow(token)
    torch_token = torch.from_numpy(token).cuda()
    return flow_token, torch_token


def align_embedding():
    flow_model = get_flow_t5_model()
    megatron_model = get_megatron_t5_model()
    load_megatron_embedding_weight(flow_model, megatron_model)
    flow_token, torch_token = get_random_token_input()

    with flow.no_grad():
        flow_output = flow_model(flow_token).numpy()
    with torch.no_grad():
        megatron_output = megatron_model(torch_token, t5_position_ids(torch_token)).cpu().numpy()
    
    print("max diff: ", np.max(np.abs(flow_output - megatron_output)))


def align_encoder():
    megatron_encoder = get_megatron_t5_encoder_layer()
    




if __name__ == '__main__':
    from utils import get_sample
    tokens_enc, _, enc_mask, _, _ = get_sample(mode='flow')
    enc_mask = enc_mask.unsqueeze(1)

    # init flow
    from t5_model_layers.t5_model_embedding import T5Embedding
    from t5_model_layers.t5_model_encoder import T5EncoderLayer
    flow_embedding = T5Embedding(HIDDEN_SIZE, VOCAB_SIZE, ENCODER_SEQ_LENGTH, 0.1).eval()
    flow_encoder_layers = [
        T5EncoderLayer(HIDDEN_SIZE, FFN_HIDDEN_SIZE, NUM_ATTENTION_HEADS, flow.nn.init.xavier_normal_, 0).eval()
    for i in range(12)]

    # init megatron
    from get_megatron_t5 import get_t5_model
    model = get_t5_model()
    megatron_embedding = model.language_model.embedding
    megatron_encoder_layers = model.language_model.encoder.layers

    # load weight
    from utils import load_megatron_embedding_weight, load_megatron_encoder_layer_weight
    load_megatron_embedding_weight(flow_embedding, megatron_embedding)
    for i in range(12):
        load_megatron_encoder_layer_weight(flow_encoder_layers[i], megatron_encoder_layers[i])

    # hook
    flow_results = []
    def flow_forward_hook(module, inp, out):
        flow_results.append([module, out])
    
    def flow_forward_hook_getinput(module, inp, out):
        flow_results.append(['inp', module, inp])
    
    megatron_results = []
    def megatron_forward_hook(module, inp, out):
        megatron_results.append([module, out])
        
    def megatron_forward_hook_getinput(module, inp, out):
        megatron_results.append(['inp', module, out])


    # register hook
    flow_encoder_layers[0].layer.self_attention.query_key_value.register_forward_hook(flow_forward_hook)
    # flow_encoder_layers[0].layer.self_attention.dropout.register_forward_hook(flow_forward_hook)
    # flow_encoder_layers[0].layer.self_attention.query_key_value.register_forward_hook(flow_forward_hook_getinput)
    # flow_encoder_layers[0].layer.self_attention.query_key_value.register_forward_hook(flow_forward_hook)


    megatron_encoder_layers[0].self_attention.query_key_value.register_forward_hook(megatron_forward_hook)
    # megatron_encoder_layers[0].self_attention.query_key_value.register_forward_hook(megatron_forward_hook_getinput)
        
    # forward
    flow_tokens_enc, _, flow_enc_mask, _, _ = get_sample(mode='flow')
    flow_enc_mask = flow_enc_mask.unsqueeze(1)
    flow_embedding_output = flow_embedding(flow_tokens_enc)
    flow_hidden_state = flow_embedding_output
    flow_hidden_states = []
    for i in range(12):
        flow_hidden_state = flow_encoder_layers[i](flow_hidden_state, flow_enc_mask)
        flow_hidden_states.append(flow_hidden_state)

    tokens_enc, _, enc_mask, _, _ = get_sample(mode='torch')
    position_ids = t5_position_ids(tokens_enc)
    with torch.no_grad():
        megatron_embedding_output = megatron_embedding(tokens_enc, position_ids)
        megatron_hidden_state = megatron_embedding_output.transpose(0, 1)
        enc_mask = enc_mask.unsqueeze(1)
        megatron_hidden_states = []
        for i in range(12):
            megatron_hidden_state = megatron_encoder_layers[i](megatron_hidden_state, enc_mask)
            megatron_hidden_states.append(megatron_hidden_state.transpose(0, 1))
    
    

    print('max embedding diff: ', np.max(np.abs(flow_embedding_output.numpy() - megatron_embedding_output.cpu().numpy())))
    for i in range(12):
        print(f'max encoder layers{i} diff: ', np.max(np.abs(flow_hidden_states[i].numpy() - megatron_hidden_states[i].detach().cpu().numpy())))