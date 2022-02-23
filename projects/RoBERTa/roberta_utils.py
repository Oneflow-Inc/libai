import oneflow as flow
import oneflow.nn as nn
import inspect
from typing import Callable, List, Set, Tuple

from .dev_ops import LayerNorm

# for cumsum
import numpy as np


def init_weights(module):

    if isinstance(module, nn.Linear):
        # module.weight.data.normal_(mean=0.0, std=0.02)
        module.weight.data.fill_(0.01)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Embedding):
        # module.weight.data.normal_(mean=0.0, std=0.02)
        module.weight.data.fill_(0.01)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].fill_(0.01)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.fill_(0.0)
        module.weight.data.fill_(1.0)
    elif isinstance(module, LayerNorm):
        module.b_2.data.fill_(0.0)
        module.a_2.data.fill_(1.0)


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):

    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).to(flow.int32)
    # oneflow does not support cumsum now.
    mask_cumsum = flow.tensor(np.cumsum(mask.numpy(), axis=1)).to(
        mask.device, mask.dtype)
    incremental_indices = (mask_cumsum + past_key_values_length) * mask
    return incremental_indices.to(flow.int64) + padding_idx


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], flow.Tensor]:

    mask = flow.ones(n_heads, head_size)
    # Convert to set and remove already pruned heads
    heads = set(heads) - already_pruned_heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: flow.Tensor = flow.arange(len(mask), dtype=flow.int64)[mask]
    return heads, index


def prune_linear_layer(layer: nn.Linear, index: flow.Tensor, dim: int = 0) -> nn.Linear:

    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(
        new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def apply_chunking_to_forward(
    forward_fn: Callable[..., flow.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> flow.Tensor:

    assert len(
        input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"
    tensor_shape = input_tensors[0].shape[chunk_dim]

    assert all(
        input_tensor.shape[chunk_dim] == tensor_shape for input_tensor in input_tensors
    ), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(
        inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(
            num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk)
                              for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return flow.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)

# replace flow.einsum when
# position_embedding_type == "relative_key" or "relative_key_query"


def position_scores(layer, embed):

    assert layer.dim() == 4
    assert embed.dim() == 3
    assert layer.shape[3] == embed.shape[2]
    assert layer.shape[2] == embed.shape[0]
    b, h, l, d = layer.shape
    l, r, d = embed.shape

    layer = layer.unsqueeze(-2)
    embed = embed.transpose(-2, -
                            1).unsqueeze(0).unsqueeze(0).expand(b, h, l, d, r)

    return flow.matmul(layer, embed).squeeze(-2)
