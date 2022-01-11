import oneflow as flow
import numpy as np
import torch

from libai.utils import distributed as dist
from torch.nn.modules import conv


def convert_and_copy_tensor(tensor_lhs: flow.Tensor, tensor_rhs: torch.Tensor):
    """ copy torch tensor weights to flow tensor weights

    Args:
        tensor_lhs (flow.Tensor)
        tensor_rhs (torch.Tensor)
    """
    tensor_rhs = flow.Tensor(tensor_rhs.cpu().float().numpy())
    tensor_rhs = flow.to_consistent(tensor_rhs, placement=tensor_lhs.placement, sbp=tensor_lhs.sbp)
    tensor_lhs.copy_(tensor_rhs)


def numpy_to_flow(tensor: np.ndarray) -> flow.Tensor:
    if tensor.dtype == np.bool:
        tensor = tensor.astype('int64')
    tensor = flow.from_numpy(tensor)
    tensor = tensor.to_consistent(sbp=flow.sbp.broadcast, placement=dist.get_layer_placement(0))
    return tensor

def get_sample(mode: str):
    from pathlib import Path
    data_path = Path("/home/wang/data/t5/samples")
    tokens_enc = np.load(data_path / 'tokens_enc.npy')[0: 1]
    tokens_dec = np.load(data_path / 'tokens_dec.npy')[0: 1]
    enc_mask = np.load(data_path / 'enc_mask.npy')[0: 1]
    dec_mask = np.load(data_path / 'dec_mask.npy')[0: 1]
    enc_dec_mask = np.load(data_path / 'enc_dec_mask.npy')[0: 1]
    if mode == 'flow':
        func = numpy_to_flow
    elif mode == 'torch':
        func = lambda x: torch.from_numpy(x).cuda()
    
    tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = map(
        func, [tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask])
    return tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask


def load_megatron_embedding_weight(flow_model: flow.nn.Module, torch_model: torch.nn.Module):
    flow_state = flow_model.state_dict()
    torch_state = torch_model.state_dict()

    # embedding
    # vocab_embeddings
    vocab_embeddings_weight = torch_state['word_embeddings.weight']
    convert_and_copy_tensor(flow_state['vocab_embeddings.weight'], vocab_embeddings_weight)
    # position embeddings
    position_embeddings_weight = torch_state['position_embeddings.weight']
    convert_and_copy_tensor(flow_state['position_embeddings.weight'], position_embeddings_weight)
    if 'tokentype_embeddings.weight' in torch_state and 'tokentype_embeddings.weight' in flow_state:
        tokentype_embeddings_weight = torch_state['tokentype_embeddings.weight']
        convert_and_copy_tensor(flow_state['tokentype_embeddings.weight'], tokentype_embeddings_weight)


def load_megatron_encoder_layer_weight(flow_model: flow.nn.Module, torch_model: torch.nn.Module):
    '''
    megatron named parameters:
    input_layernorm.weight
    input_layernorm.bias
    self_attention.query_key_value.weight
    self_attention.query_key_value.bias
    self_attention.dense.weight
    self_attention.dense.bias
    post_attention_layernorm.weight
    post_attention_layernorm.bias
    mlp.dense_h_to_4h.weight
    mlp.dense_h_to_4h.bias
    mlp.dense_4h_to_h.weight
    mlp.dense_4h_to_h.bias
    '''
    flow_state = flow_model.state_dict()
    torch_state = torch_model.state_dict()
    for k, v in torch_state.items():
        if 'weight' in k and len(v.shape) == 2:
            v = v.transpose(0, 1)
        convert_and_copy_tensor(flow_state['layer.' + k], v)