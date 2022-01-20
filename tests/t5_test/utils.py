import oneflow as flow
import numpy as np
import torch

from libai.utils import distributed as dist


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
    func = {
        'flow': numpy_to_flow,
        'torch': lambda x: torch.from_numpy(x).cuda(),
        'numpy': lambda x: x,
    }
    if mode == 'flow':
        enc_mask, dec_mask, enc_dec_mask = map(lambda x: ~x, [enc_mask, dec_mask, enc_dec_mask])
    
    tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = map(
        func[mode], [tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask])
    return tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask


def get_random_sample(vocab_size):
    tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = get_sample('numpy')
    tokens_enc = np.random.randint(0, vocab_size, size=tokens_enc.shape)
    tokens_dec = np.random.randint(0, vocab_size, size=tokens_dec.shape)
    flow_tensors = tuple(map(numpy_to_flow, [tokens_enc, tokens_dec, ~enc_mask, ~dec_mask, ~enc_dec_mask]))
    torch_tensors = tuple(map(
        lambda x: torch.from_numpy(x).cuda(), 
        [tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask]
    ))
    return flow_tensors, torch_tensors


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


def load_megatron_decoder_layer_weight(flow_model: flow.nn.Module, torch_model: torch.nn.Module):
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
    inter_attention.query.weight
    inter_attention.query.bias
    inter_attention.key_value.weight
    inter_attention.key_value.bias
    inter_attention.dense.weight
    inter_attention.dense.bias
    post_inter_attention_layernorm.weight
    post_inter_attention_layernorm.bias
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
        if 'inter' in k:
            k = k.replace('inter', 'cross')
        convert_and_copy_tensor(flow_state['layer.' + k], v)

def load_megatron_encoder_weight(flow_model: flow.nn.Module, torch_model: torch.nn.Module):
    flow_state = flow_model.state_dict()
    torch_state = torch_model.state_dict()

    for k, v in torch_state.items():
        if 'weight' in k and len(v.shape) == 2:
            v = v.transpose(0, 1)
        if 'inter' in k:
            k = k.replace('inter', 'cross')
        convert_and_copy_tensor(flow_state[k], v)


def load_megatron_weight(flow_model: flow.nn.Module, torch_model: torch.nn.Module):
    '''
    embedding.vocab_embeddings.weight
    embedding.position_embeddings.weight
    embedding.tokentype_embedding.weight
    encoders.0.input_layernorm.weight
    encoders.11.mlp.dense_h_to_4h.weight
    encoders.11.mlp.dense_h_to_4h.bias
    encoders.11.mlp.dense_4h_to_h.weight
    encoders.11.mlp.dense_4h_to_h.bias
    encoder_final_layernorm.weight
    encoder_final_layernorm.bias
    decoders.0.input_layernorm.weight
    decoders.0.input_layernorm.bias
    decoders.0.self_attention.query_key_value.weight
    decoders.0.self_attention.query_key_value.bias
    decoders.0.self_attention.dense.weight

    language_model.embedding.word_embeddings.weight
    language_model.embedding.position_embeddings.weight
    language_model.encoder.layers.0.input_layernorm.weight
    language_model.decoder.layers.11.inter_attention.dense.bias
    language_model.decoder.layers.11.post_inter_attention_layernorm.weight
    language_model.decoder.layers.11.post_inter_attention_layernorm.bias
    language_model.decoder.layers.11.mlp.dense_h_to_4h.weight
    language_model.decoder.layers.11.mlp.dense_h_to_4h.bias
    language_model.decoder.layers.11.mlp.dense_4h_to_h.weight
    language_model.decoder.layers.11.mlp.dense_4h_to_h.bias
    language_model.decoder.final_layernorm.weight
    language_model.decoder.final_layernorm.bias
    lm_head.bias

    '''
    flow_state = flow_model.state_dict()
    torch_state = torch_model.state_dict()

    print('flow_weight nums: ', len(flow_model.state_dict()))
    print('torch_weight nums: ', len(torch_model.state_dict()))

    used_flow_keys = set()
    used_torch_keys = set()

    for torch_k, v in torch_state.items():
        k = torch_k
        if 'weight'in k and len(v.shape) == 2 and 'embedding.' not in k:
            v = v.transpose(0, 1)
        if 'inter' in k:
            k = k.replace('inter', 'cross')
        if k.startswith('language_model.'):
            k = k[len('language_model.'): ]
        convert_and_copy_tensor(flow_state[k], v)
        used_flow_keys.add(k)
        used_torch_keys.add(torch_k)

    print(set(flow_state.keys()) - used_flow_keys)
    print(set(torch_state.keys()) - used_torch_keys)
        