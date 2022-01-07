import oneflow as flow
import torch
import numpy as np

from t5_model_layers.t5_model_embedding import T5Embedding
from get_megatron_t5 import get_t5_model
from megatron.model.t5_model import T5Model, t5_position_ids 
from utils import convert_and_copy_tensor, numpy_to_flow, load_megatron_embedding_weight

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

def get_flow_t5_model():
    return T5Embedding(HIDDEN_SIZE, VOCAB_SIZE, ENCODER_SEQ_LENGTH, 0.1).eval()


def get_megatron_t5_model():
    model = get_t5_model()
    model = model.language_model.embedding
    return model


def get_random_token_input():
    token = np.random.randint(0, VOCAB_SIZE, size=(1, ENCODER_SEQ_LENGTH), dtype=np.int64)
    flow_token = numpy_to_flow(token)
    torch_token = torch.from_numpy(token).cuda()
    return flow_token, torch_token

if __name__ == '__main__':
    flow_model = get_flow_t5_model()
    megatron_model = get_megatron_t5_model()
    load_megatron_embedding_weight(flow_model, megatron_model)
    flow_token, torch_token = get_random_token_input()

    with flow.no_grad():
        flow_output = flow_model(flow_token).numpy()
    with torch.no_grad():
        megatron_output = megatron_model(torch_token, t5_position_ids(torch_token)).cpu().numpy()
    
    print("max diff: ", np.max(np.abs(flow_output - megatron_output)))