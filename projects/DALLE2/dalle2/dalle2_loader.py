import logging

import oneflow as flow
from oneflow.framework.check_point_v2 import _broadcast_py_object

import libai.utils.distributed as dist
from libai.models.build import build_model
from libai.models.utils.model_loader.base_loader import (
    ModelLoaderHuggerFace,
    _load_state_dict_into_model,
)

logger = logging.getLogger("libai.dalle2." + __name__)


class Dalle2ModelLoader(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_1 = ""
        self.base_model_prefix_2 = ""

    def _convert_state_dict(self, state_dict, module="prior"):
        old_keys = []
        new_keys = []
        if module == "prior":
            for k in state_dict.keys():
                if "clip." in k:
                    continue
                old_keys.append(k)
                if k.endswith(".g"):
                    k = k[:-1] + "weight"
                elif k.startswith("net.causal_transformer"):
                    if k.endswith("gamma"):
                        k = k[:-5] + "weight"
                    elif k.endswith("beta"):
                        k = k[:-4] + "bias"
                new_keys.append("prior." + k)
        elif module == "decoder":
            for k in state_dict.keys():
                if "clip." in k:
                    continue
                old_keys.append(k)
                if k.endswith(".g"):
                    k = k[:-1] + "weight"
                elif "cross_attn" in k:
                    if k.endswith("gamma"):
                        k = k[:-5] + "weight"
                    elif k.endswith("beta"):
                        k = k[:-4] + "bias"
                new_keys.append("decoder." + k)
        ret_state_dict = {}
        for old_key, new_key in zip(old_keys, new_keys):
            ret_state_dict[new_key] = state_dict.pop(old_key)
        return ret_state_dict

    def load(self):
        if dist.is_main_process():
            # prior
            logger.info("loading torch model prior...")
            torch_state_dict = self._load_torch_state_dict(self.libai_cfg.model.prior_weight_path)[
                "ema_model"
            ]
            logger.info("converting torch model prior into oneflow model...")
            flow_state_dict = self._convert_tensors(torch_state_dict)
            prior_state_dict = self._convert_state_dict(flow_state_dict)
            # decoder
            logger.info("loading torch model decoder...")
            torch_state_dict = self._load_torch_state_dict(self.libai_cfg.model.decoder_weight_path)
            flow_state_dict = self._convert_tensors(torch_state_dict)
            logger.info("converting torch model decoder into oneflow model...")
            decoder_state_dict = self._convert_state_dict(flow_state_dict, module="decoder")
            flow_state_dict = {**prior_state_dict, **decoder_state_dict}
        else:
            flow_state_dict = None

        logger.info("building LiBai model...")
        self.libai_cfg = _broadcast_py_object(self.libai_cfg, src=0)
        self.model = build_model(self.model)
        self.model._apply(dist.convert_to_distributed_default_setting)
        self.model = self.model.eval()

        flow.cuda.empty_cache()
        # State_dict to global
        logger.info("transfering state_dict local to global...")
        flow_state_dict = self._state_dict_to_global(flow_state_dict, mode="pytorch")  # oom
        # Load
        # (
        #     model,
        #     missing_keys,
        #     unexpected_keys,
        #     mismatched_keys,
        #     error_msgs,
        # ) = self._load_pretrained_model(self.model, flow_state_dict, self.pretrained_model_path)
        logger.info("loading model weights into LiBai...")
        _load_state_dict_into_model(self.model, flow_state_dict, "")
        return self.model
