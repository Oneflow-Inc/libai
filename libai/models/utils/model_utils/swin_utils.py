import json

import oneflow as flow

from libai.models import SwinTransformer

from .base_utils import LoadPretrainedBase


class LoadPretrainedSwin(LoadPretrainedBase):
    def __init__(self, model, default_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, default_cfg, pretrained_model_path, **kwargs)
        
        """NOTE: base_model_prefix_1 is SWIN's prefix in Transformers.
        base_model_prefix_2 is SWIN's prefix in LiBai."""
    
        self.base_model_prefix_1 = "swin"
        self.base_model_prefix_2 = "swin"
        
    def _convert_state_dict(self, flow_state_dict, cfg):
        """Convert state_dict's keys to match model.

        Args:
            flow_state_dict (OrderedDict): model state dict.
            cfg (dict): model's default config dict.

        Returns:
            OrderedDict: flow state dict.
        """
        # The converted checkpoint.
        oneflow_state_dict = flow_state_dict.copy()

        # Get configs tiny swin
        num_heads = cfg.get("num_heads")     # 3 6 12 24
        embed_dim = [cfg.get("embed_dim") * 2**i for i in range(0, 4)]  # 96 192 384 768
        
        head_size = [int(dim/head) for dim, head in  zip(embed_dim, num_heads)] # 32 32 32 32

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)

        prefix = "swin." if has_prefix else ""
        index_idx_1 = 3 if has_prefix else 2
        index_idx_2 = 5 if has_prefix else 4
        # qkv_idx = 7
        
        old_keys = oneflow_state_dict.keys()

        for key in list(old_keys):

            # Convert swin's embedding layers
            if "embeddings" in key:
                if "patch_embeddings.projection" in key:
                    if "weight" in key:
                        new_key = "patch_embed.proj.weight"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif "bias" in key:
                        new_key = "patch_embed.proj.bias"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "norm" in key:
                    if "weight" in key:
                        new_key = "patch_embed.norm.weight"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif "bias" in key:
                        new_key = "patch_embed.norm.bias"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                
            # Convert swin's layernorm layers
            elif "layernorm_before" in key:
                index_layer = key.split(".")[index_idx_1]
                index_block = key.split(".")[index_idx_2]
                if "weight" in key:
                    new_key = "layers." + index_layer + ".blocks." + index_block + ".norm1.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "layers." + index_layer + ".blocks." + index_block + ".norm1.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                
            elif "layernorm_after" in key:
                index_layer = key.split(".")[index_idx_1]
                index_block = key.split(".")[index_idx_2]
                if "weight" in key:
                    new_key = "layers." + index_layer + ".blocks." + index_block + ".norm2.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "layers." + index_layer + ".blocks." + index_block + ".norm2.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    
            # Convert swin's attention layers
            elif "attention" in key:
                index_layer = key.split(".")[index_idx_1]
                index_block = key.split(".")[index_idx_2]
                if "self" in key:
                    if "relative_position_bias_table" in key:  # convert relative_position_bias_table/index
                        new_key = "layers." + index_layer + ".blocks." + index_block + ".attn.relative_position_bias_table"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif "relative_position_index" in key:
                        new_key = "layers." + index_layer + ".blocks." + index_block + ".attn.relative_position_index"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    else:   # qkv
                        if (
                            "layers." + index_layer + ".blocks." + index_block + ".attn.qkv.weight" in oneflow_state_dict.keys()
                        ):
                            continue
                        q_w = key
                        k_w = q_w.replace("query", "key")
                        v_w = q_w.replace("query", "value")
                        q_b = q_w.replace("weight", "bias")
                        k_b = k_w.replace("weight", "bias")
                        v_b = v_w.replace("weight", "bias")

                        qkv_w = flow.cat(
                            (
                                oneflow_state_dict.pop(q_w),
                                oneflow_state_dict.pop(k_w),
                                oneflow_state_dict.pop(v_w),
                            ),
                            dim=0,
                        )
                        qkv_b = flow.cat(
                            (
                                oneflow_state_dict.pop(q_b),
                                oneflow_state_dict.pop(k_b),
                                oneflow_state_dict.pop(v_b),
                            ),
                            dim=-1,
                        )

                        qkv_w = self._fix_qkv_ordering(qkv_w, head_size[int(index_layer)], num_heads[int(index_layer)])
                        qkv_b = self._fix_qkv_ordering(qkv_b, head_size[int(index_layer)], num_heads[int(index_layer)])

                        new_key = (
                           "layers."  + index_layer + ".blocks." + index_block + ".attn.qkv.weight"
                        )
                        oneflow_state_dict[new_key] = qkv_w
                        
                        new_key = new_key.replace("weight", "bias")
                        oneflow_state_dict[new_key] = qkv_b
                    
                elif "output" in key:
                    if "dense" in key:
                        if "weight" in key:
                            new_key = "layers."  + index_layer + ".blocks." + index_block + ".attn.proj.weight"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        if "bias" in key:
                            new_key = "layers."  + index_layer + ".blocks." + index_block + ".attn.proj.bias"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                
            elif "intermediate" in key:
                index_layer = key.split(".")[index_idx_1]
                index_block = key.split(".")[index_idx_2]
                if "weight" in key:
                    if (
                        "layers." + index_layer + ".blocks." + index_block + ".mlp.dense_h_to_4h.weight" in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = "layers." + index_layer + ".blocks." + index_block + ".mlp.dense_h_to_4h.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                    
            elif "output" in key:
                index_layer = key.split(".")[index_idx_1]
                index_block = key.split(".")[index_idx_2]
                if "dense.weight" in key:
                    if (
                        "layers." + index_layer + ".blocks." + index_block + ".mlp.dense_4h_to_h.weight" in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    new_key = "layers." + index_layer + ".blocks." + index_block + ".mlp.dense_4h_to_h.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
            
            elif "downsample" in key:
                index_layer = key.split(".")[index_idx_1]
                if "reduction.weight" in key:
                    new_key = "layers." + index_layer + ".downsample.reduction.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "norm" in key:
                    if(
                        "layers." + index_layer + ".downsample.norm.weight" in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    new_key = "layers." + index_layer + ".downsample.norm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                
            elif "layernorm" in key:
                if "weight" in key:
                    new_key = "norm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "norm.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
            elif "classifier" in key:
                if "weight" in key:
                    new_key = "head.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "head.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
            else:
                oneflow_state_dict[key] = oneflow_state_dict.pop(key)
                
        return oneflow_state_dict
    
    
    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.

        Args:
            config_file (str): Path of config file.
        """
        with open(config_file, mode="r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # update default_cfg by config.json
        self.default_cfg.img_size = cfg_dict["image_size"]
        self.default_cfg.patch_size = cfg_dict["patch_size"]
        self.default_cfg.embed_dim = cfg_dict["embed_dim"]
        self.default_cfg.depths = cfg_dict["depths"]
        self.default_cfg.num_heads = cfg_dict["num_heads"]
        self.default_cfg.window_size = cfg_dict["window_size"]
        self.default_cfg.mlp_ratio = cfg_dict["mlp_ratio"]
        self.default_cfg.qkv_bias = cfg_dict["qkv_bias"]
        self.default_cfg.drop_path_rate = cfg_dict["drop_path_rate"]
        
        
        # update default_cfg by kwargs
        for k, v in self.kwargs.items():
            self.default_cfg[k] = v

if __name__ == "__main__":
    model = SwinTransformer()
    print(model)