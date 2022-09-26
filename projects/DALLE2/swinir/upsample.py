import os

import oneflow as flow
import requests

from .models import SwinIR as net


def load_torch_weight(model, model_path):
    # load torch weight
    import torch

    param_key_g = "params_ema"
    pretrained_model = torch.load(model_path, map_location="cpu")
    pretrained_model = (
        pretrained_model[param_key_g]
        if param_key_g in pretrained_model.keys()
        else pretrained_model
    )
    new_state_dict = {}
    for k, v in pretrained_model.items():
        flow_tensor = flow.tensor(v.numpy())
        new_state_dict[k] = flow_tensor
    model.load_state_dict(new_state_dict, strict=True)
    return model


def load_model(model_path=None):
    # set up model
    if os.path.exists(model_path):
        print(f"loading model from {model_path}")
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}".format(
            os.path.basename(model_path)
        )
        r = requests.get(url, allow_redirects=True)
        print(f"downloading model {model_path}")
        open(model_path, "wb").write(r.content)
    model = net(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="3conv",
    )
    model = load_torch_weight(model, model_path)
    return model


def upsample4x(img_lq, model):
    """upsample img from h*w to (4h) * (4w)"""
    device = flow.device("cuda" if flow.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)
    img_lq = img_lq.to(device)

    window_size = 8
    scale = 4

    # inference
    with flow.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = flow.cat([img_lq, flow.flip(img_lq, [2])], 2)[:, :, : h_old + h_pad, :]
        img_lq = flow.cat([img_lq, flow.flip(img_lq, [3])], 3)[:, :, :, : w_old + w_pad]
        output = model(img_lq)
        output = output[..., : h_old * scale, : w_old * scale]
    output = output.clamp_(0, 1)
    return output


def upsample16x(imgs, model):
    return upsample4x(upsample4x(imgs, model), model)
