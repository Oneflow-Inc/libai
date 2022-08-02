import oneflow as flow
import oneflow.nn as nn


def load_model(basic_path, model: flow.nn.Module):
    """
    Args:
        basic_path: pytorch pre-training file path
        model: oneflow model,type: flow.nn.Module()
    References:
        This file is used to import the pytorch version of swinv2 pre-training weights
    Returns:
        None
    """
    import torch

    model_dict = torch.load(basic_path)
    model_state_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in model_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            if model_state_dict[k].is_global:
                pretrained_dict[k] = convert_tensor(v).to_global(
                    placement=model_state_dict[k].placement, sbp=model_state_dict[k].sbp
                )
            else:
                pretrained_dict[k] = convert_tensor(v)
        elif (
            k.replace("fc1", "dense_h_to_4h") in model_state_dict
            and v.shape == model_state_dict[k.replace("fc1", "dense_h_to_4h")].shape
        ):
            continue
            copy_k = k.replace("fc1", "dense_h_to_4h")
            if model_state_dict[copy_k].is_global:
                pretrained_dict[copy_k] = convert_tensor(v).to_global(
                    placement=model_state_dict[copy_k].placement, sbp=model_state_dict[copy_k].sbp
                )
            else:
                pretrained_dict[copy_k] = convert_tensor(v)
        elif (
            k.replace("fc2", "dense_4h_to_h") in model_state_dict
            and v.shape == model_state_dict[k.replace("fc2", "dense_4h_to_h")].shape
        ):
            continue
            copy_k = k.replace("fc2", "dense_4h_to_h")
            if model_state_dict[copy_k].is_global:
                pretrained_dict[copy_k] = convert_tensor(v).to_global(
                    placement=model_state_dict[copy_k].placement, sbp=model_state_dict[copy_k].sbp
                )
            else:
                pretrained_dict[copy_k] = convert_tensor(v)
        else:
            raise NotImplementedError

    pretrained_key = list(pretrained_dict.keys())
    print("load_keys:", pretrained_key)
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    return model
