import omegaconf
import oneflow as flow
from oneflow.framework.check_point_v2 import _broadcast_py_object
import logging

import libai.utils.distributed as dist
from libai.models.build import build_model
from libai.models.utils.model_utils.base_loader import ModelLoaderHuggerFace

class Dalle2ModelLoader(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_1 = "transformer"
        self.base_model_prefix_2 = "dalle2"

    def _convert_state_dict(self, state_dict, module = 'prior'):
        old_keys = []
        new_keys = []
        if module == 'prior':
            for k in state_dict.keys():
                if "clip." in k:
                    continue
                old_keys.append(k)
                if k.endswith(".g"):
                    k = k[:-1] + "weight"
                elif k.startswith("net.causal_transformer"):
                    if k.endswith("gamma"):
                        k = k[:-5] + 'weight'
                    elif k.endswith('beta'):
                        k = k[:-4] + 'bias'
                new_keys.append("prior." + k)
        elif module == 'decoder':
            for k in state_dict.keys():
                if 'clip.' in k:
                    continue
                old_keys.append(k)
                if k.endswith(".g"):
                    k = k[:-1] + "weight"
                elif 'cross_attn' in k:
                    if k.endswith('gamma'):
                        k = k[:-5] + "weight"
                    elif k.endswith('beta'):
                        k = k[:-4] + "bias"
                new_keys.append("decoder." + k)
        ret_state_dict = {}
        for old_key, new_key in zip(old_keys, new_keys):
            ret_state_dict[new_key] = state_dict.pop(old_key)
        return ret_state_dict


    def load(self):
        if dist.is_main_process():
            #prior
            torch_state_dict = self._load_torch_state_dict(self.libai_cfg.model.prior_weight_path)['ema_model']
            flow_state_dict = self._convert_tensors(torch_state_dict)
            prior_state_dict = self._convert_state_dict(flow_state_dict)
            #decoder
            torch_state_dict = self._load_torch_state_dict(self.libai_cfg.model.decoder_weight_path)
            flow_state_dict = self._convert_tensors(torch_state_dict)
            decoder_state_dict = self._convert_state_dict(flow_state_dict, module='decoder')
            flow_state_dict = {**prior_state_dict, ** decoder_state_dict}
        else:
            flow_state_dict = None

        self.libai_cfg = _broadcast_py_object(self.libai_cfg, src=0)
        self.model = build_model(self.model)
        # State_dict to global
        flow_state_dict = self._state_dict_to_global(flow_state_dict, mode="pytorch")
        # Load
        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            error_msgs,
        ) = self._load_pretrained_model(self.model, flow_state_dict, self.pretrained_model_path)

        return model
        

    def load_prior_weight(self, prior, prior_weight_path):
        if isinstance(prior, omegaconf.dictconfig.DictConfig):
            prior = build_model(prior)
        import torch
        state_dict = torch.load(prior_weight_path, map_location="cpu")['ema_model']
        for k, torch_tensor in state_dict.items():
            if "clip." in k:
                continue
            if k.endswith(".g"):
                k = k[:-1] + "weight"
            elif k.startswith("net.causal_transformer"):
                if k.endswith("gamma"):
                    k = k[:-5] + 'weight'
                elif k.endswith('beta'):
                    k = k[:-4] + 'bias'
            assert k in prior.state_dict(), k
            flow_tensor = flow.tensor(torch_tensor.cpu().numpy(), placement=prior.state_dict()[
                                      k].placement, sbp=prior.state_dict()[k].sbp)
            prior.state_dict()[k].data.copy_(flow_tensor.data)

        return prior.eval()

    def load_decoder_weight(self, decoder, decoder_weight_path):
        if isinstance(decoder, omegaconf.dictconfig.DictConfig):
            decoder = build_model(decoder)
        import torch
        state_dict = torch.load(decoder_weight_path, map_location="cpu")
        for k, torch_tensor in state_dict.items():
            if 'clip.' in k:
                continue
            if k.endswith(".g"):
                k = k[:-1] + "weight"
            elif 'cross_attn' in k:
                if k.endswith('gamma'):
                    k = k[:-5] + "weight"
                elif k.endswith('beta'):
                    k = k[:-4] + "bias"
            assert k in decoder.state_dict().keys(), k
            flow_tensor = flow.tensor(torch_tensor.cpu().numpy(), placement=decoder.state_dict()[
                                      k].placement, sbp=decoder.state_dict()[k].sbp)
            decoder.state_dict()[k].data.copy_(flow_tensor.data)
        return decoder.eval()
