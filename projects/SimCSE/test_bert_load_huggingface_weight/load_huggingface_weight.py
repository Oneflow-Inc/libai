from collections import OrderedDict

import oneflow as flow
import torch


def convert_tensor(tensor):
    """Convert pytorch_tensor to oneflow_tensor

    Args:
        tensor (torch.tensor): Weight of models' parameters in pytorch_model.bin

    Returns:
        oneflow.tensor
    """
    tensor = tensor.float()
    return flow.Tensor(tensor.cpu().numpy())


def convert_state_dict(state, layers, hidden_size, num_heads, head_size):
    """Convert pytorch_tensor to oneflow_tensor and save as a state_dict

    Args:
        state (OrderedDict): State_dict of Pytorch model.
        layers (int): BERT's number of hidden layers.
        hidden_size (int): The hidden_size of BERT.
        num_heads (int): The num_head of BERT.
        head_size (int): The Head_size of BERT.

    Returns:
        OrderedDict: State_dict of OneFlow model.
    """
    save = OrderedDict()
    not_saved = []
    Layers = layers
    for name, tensor in state.items():
        if "embeddings" in name:
            if "word_embeddings" in name:
                save["embeddings.vocab_embeddings.weight"] = convert_tensor(tensor)
            elif "position_embeddings" in name:
                save["embeddings.position_embeddings.weight"] = convert_tensor(tensor)
            elif "token_type_embeddings" in name:
                save["embeddings.tokentype_embeddings.weight"] = convert_tensor(tensor)
            elif "LayerNorm.gamma" in name:
                save["encoders.0.input_layernorm.weight"] = convert_tensor(tensor)
            elif "LayerNorm.beta" in name:
                save["encoders.0.input_layernorm.bias"] = convert_tensor(tensor)

        elif "attention" in name:
            if "self" in name:
                index = name.split(".")[3]
                if "encoders." + index + ".self_attention.query_key_value.weight" in save.keys():
                    continue
                q_w = name.replace(name.split(".")[6], "query").replace(
                    name.split(".")[7], "weight"
                )
                k_w = name.replace(name.split(".")[6], "key").replace(name.split(".")[7], "weight")
                v_w = name.replace(name.split(".")[6], "value").replace(
                    name.split(".")[7], "weight"
                )
                q_b = name.replace(name.split(".")[6], "query").replace(name.split(".")[7], "bias")
                k_b = name.replace(name.split(".")[6], "key").replace(name.split(".")[7], "bias")
                v_b = name.replace(name.split(".")[6], "value").replace(name.split(".")[7], "bias")

                qkv_w = torch.cat((state[q_w], state[k_w], state[v_w]), dim=0)  # 【768*3， 768】

                # Rearrange the loaded weights for weight, you can refer:
                # https://libai.readthedocs.io/en/latest/notes/How_to_implement_huggingface%27s_weights_in_LiBai.html
                qkv_w = qkv_w.view([3, num_heads, head_size, hidden_size])
                qkv_w = qkv_w.permute(1, 0, 2, 3).contiguous().view(3 * hidden_size, hidden_size)

                qkv_b = torch.cat((state[q_b], state[k_b], state[v_b]), dim=-1)

                # # Rearrange the loaded weights for bias, you can refer:
                # https://libai.readthedocs.io/en/latest/notes/How_to_implement_huggingface%27s_weights_in_LiBai.html
                qkv_b = qkv_b.view(3, num_heads, head_size)
                qkv_b = qkv_b.permute(1, 0, 2).contiguous().view(-1)

                target_w = "encoders." + index + ".self_attention.query_key_value.weight"
                save[target_w] = convert_tensor(qkv_w)
                target_b = "encoders." + index + ".self_attention.query_key_value.bias"
                save[target_b] = convert_tensor(qkv_b)
            elif "output" in name:
                index = name.split(".")[3]
                if "dense" in name:
                    if "weight" in name:
                        target = "encoders." + index + ".self_attention.dense.weight"
                        save[target] = convert_tensor(tensor)
                    elif "bias" in name:
                        target = "encoders." + index + ".self_attention.dense.bias"
                        save[target] = convert_tensor(tensor)
                elif "LayerNorm" in name:
                    if "gamma" in name:
                        target = "encoders." + index + ".post_attention_layernorm.weight"
                        save[target] = convert_tensor(tensor)
                    elif "beta" in name:
                        target = "encoders." + index + ".post_attention_layernorm.bias"
                        save[target] = convert_tensor(tensor)

        elif "intermediate" in name:
            index = name.split(".")[3]
            if "encoders." + index + ".mlp.dense_h_to_4h.weight" in save.keys():
                continue
            w = "bert.encoder.layer." + index + ".intermediate.dense.weight"
            b = "bert.encoder.layer." + index + ".intermediate.dense.bias"
            t_w = "encoders." + index + ".mlp.dense_h_to_4h.weight"
            t_b = "encoders." + index + ".mlp.dense_h_to_4h.bias"
            save[t_w] = convert_tensor(state[w])
            save[t_b] = convert_tensor(state[b])

        elif "output" in name:
            index = name.split(".")[3]
            if "dense.weight" in name:
                target = "encoders." + index + ".mlp.dense_4h_to_h.weight"
                save[target] = convert_tensor(tensor)
            elif "dense.bias" in name:
                target = "encoders." + index + ".mlp.dense_4h_to_h.bias"
                save[target] = convert_tensor(tensor)
            elif "LayerNorm.gamma" in name:
                if index == str(Layers - 1):
                    target = "final_layernorm.weight"
                    save[target] = convert_tensor(tensor)
                    continue
                target = "encoders." + str(int(index) + 1) + ".input_layernorm.weight"
                save[target] = convert_tensor(tensor)
            elif "LayerNorm.beta" in name:
                if index == str(Layers - 1):
                    target = "final_layernorm.bias"
                    save[target] = convert_tensor(tensor)
                    continue
                target = "encoders." + str(int(index) + 1) + ".input_layernorm.bias"
                save[target] = convert_tensor(tensor)

        elif "pooler" in name:
            if "weight" in name:
                save["pooler.dense.weight"] = convert_tensor(tensor)
            elif "bias" in name:
                save["pooler.dense.bias"] = convert_tensor(tensor)
        else:
            not_saved.append(name)
    return save, not_saved


def load_tensor(tensor_lhs, tensor_rhs):
    """Load the tensor to BERT.

    Args:
        tensor_lhs (flow.tensor): The tensor in state_dict.
        tensor_rhs (flow.tensor): The tensor in LiBai's BERT.
    """
    tensor_rhs = flow.to_global(tensor_rhs, placement=tensor_lhs.placement, sbp=tensor_lhs.sbp)
    tensor_lhs.copy_(tensor_rhs)


def load_huggingface_bert(model, path, hidden_size, num_heads, layers=12):
    """Load Huggingface's pretrained weights in LiBai

    Args:
        model: BRET in LiBai.
        path (str): The path of pretrained_model file.
    """
    head_size = hidden_size // num_heads
    huggingface_state_dict = torch.load(path)
    of_state_dict, _ = convert_state_dict(
        huggingface_state_dict,
        layers=layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_size=head_size,
    )
    for key, value in of_state_dict.items():
        load_tensor(model.state_dict()[key], value)
