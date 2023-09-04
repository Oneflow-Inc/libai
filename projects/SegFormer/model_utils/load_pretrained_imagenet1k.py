import json

import oneflow as flow

from libai.models.utils.model_utils.base_loader import ModelLoaderHuggerFace

class SegFLoaderImageNet1kPretrain(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        
        """NOTE: base_model_prefix_1 is SegF's prefix in Transformers.
        base_model_prefix_2 is SegF's prefix in LiBai."""

        self.base_model_prefix_1 = ""
        self.base_model_prefix_2 = ""
        
    def _convert_state_dict(self, flow_state_dict, cfg=None):
        """Convert official ImageNet1K pretrained state_dict's keys to match model.

        Args:
            flow_state_dict (OrderedDict): model state dict.
            cfg (dict): model's default config dict.

        Returns:
            OrderedDict: flow state dict.
        """
        # The converted checkpoint.
        oneflow_state_dict = flow_state_dict.copy()

        old_keys = oneflow_state_dict.keys()

        for key in list(old_keys):

            # Convert segformer's embedding layers
            if "patch_embed" in key:
                index_block = key[11]
                if "proj" in key:
                    if ( 
                        "patch_embeds." + str(int(index_block) - 1) + ".proj.weight" 
                        in oneflow_state_dict.keys() 
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = "patch_embeds." + str(int(index_block) - 1) + ".proj.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                elif "norm" in key:
                    if ( 
                        "patch_embeds." + str(int(index_block) - 1) + ".norm.weight"
                        in oneflow_state_dict.keys() 
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = "patch_embeds." + str(int(index_block) - 1) + ".norm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)

            # Convert segformer's layernorm layers
            elif "norm1" in key and "block" in key:
                index_block = key[5]
                index_block_ = key.split(".")[1]
                if "weight" in key:
                    new_key = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".norm1.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".norm1.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            elif "norm2" in key and "block" in key:
                index_block = key[5]
                index_block_ = key.split(".")[1]
                if "weight" in key:
                    new_key = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".norm1.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".norm1.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            # Convert segformer's attention layers
            elif "attn" in key:
                index_block = key[5]
                index_block_ = key.split(".")[1]
                if "attn.sr" in key:
                    if (
                        "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".attn.sr.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".attn.sr.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                    
                elif "attn.norm" in key:
                    if (
                        "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".attn.norm.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".attn.norm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                    
                elif "attn.q" in key:
                    if (
                        "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".attn.q.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    q_w = key
                    kv_w = q_w.replace("q", "kv")
                    q_b = q_w.replace("weight", "bias")
                    kv_b = kv_w.replace("weight", "bias")
                    
                    kv_w = oneflow_state_dict.pop(kv_w)
                    kv_b = oneflow_state_dict.pop(kv_b)
                    
                    
                    new_key_q_w = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".attn.q.weight"
                    oneflow_state_dict[new_key_q_w] = oneflow_state_dict.pop(q_w)
                    new_key_kv_w = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".attn.kv.weight"
                    oneflow_state_dict[new_key_kv_w] = kv_w

                    new_key_q_b = new_key_q_w.replace("weight", "bias")
                    oneflow_state_dict[new_key_q_b] = oneflow_state_dict.pop(q_b)
                    new_key_kv_b = new_key_kv_w.replace("weight", "bias")
                    oneflow_state_dict[new_key_kv_b] = kv_b
                
                elif "proj" in key:
                    if "weight" in key:
                        new_key = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".attn.proj.weight"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    if "bias" in key:
                        new_key = "blocks." + str(int(index_block) - 1) + "." + index_block_  + ".attn.proj.bias"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            elif "mlp" in key:
                index_block = key[5]
                index_block_ = key.split(".")[1]
                if "fc1.weight" in key:
                    if (
                        "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".mlp.dense_h_to_4h.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".mlp.dense_h_to_4h.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                
                elif "dwconv.weight" in key:
                    if (
                        "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".mlp.dwconv.dwconv.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".mlp.dwconv.dwconv.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                    
                elif "fc2.weight" in key:
                    if (
                        "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".mlp.dense_4h_to_h.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = "blocks." + str(int(index_block) - 1) + "." + index_block_ + ".mlp.dense_4h_to_h.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                    
            elif "norm" in key:
                index_block = key[4]
                if (
                    "layer_norms." + str(int(index_block) - 1) + ".weight"
                    in oneflow_state_dict.keys()
                ):
                    continue
                w = key
                b = key.replace("weight", "bias")
                new_key = "layer_norms." + str(int(index_block) - 1) + ".weight"
                oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                new_key = new_key.replace("weight", "bias")
                oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)

        return oneflow_state_dict
    
    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.

        Args:
            config_file (str): Path of config file.
        """
        
        # update libai_cfg by kwargs
        for k, v in self.kwargs.items():
            self.libai_cfg[k] = v
