import json

import oneflow as flow

from libai.models.utils.model_utils.base_loader import ModelLoaderHuggerFace, ModelLoaderLiBai

class SegFLoaderHuggerFace(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        
        """NOTE: base_model_prefix_1 is SegF's prefix in Transformers.
        base_model_prefix_2 is SegF's prefix in LiBai."""

        self.base_model_prefix_1 = "segformer"
        self.base_model_prefix_2 = ""
        
    def _convert_state_dict(self, flow_state_dict, cfg=None):
        """Convert state_dict's keys to match model.

        Args:
            flow_state_dict (OrderedDict): model state dict.
            cfg (dict): model's default config dict.

        Returns:
            OrderedDict: flow state dict.
        """
        prefix = "segformer."
        
        # The converted checkpoint.
        oneflow_state_dict = flow_state_dict.copy()

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)

        index_idx = 3 if has_prefix else 2
        index_decode = 2

        old_keys = oneflow_state_dict.keys()

        for key in list(old_keys):

            # Convert segformer's embedding layers
            if "patch_embeddings" in key:
                index_layer = key.split('.')[index_idx]
                if "proj" in key:
                    if ( 
                        prefix + "patch_embeds." + index_layer + ".proj.weight" 
                        in oneflow_state_dict.keys() 
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = prefix + "patch_embeds." + index_layer + ".proj.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                elif "layer_norm" in key:
                    if ( 
                        prefix + "patch_embeds." + index_layer + ".norm.weight"
                        in oneflow_state_dict.keys() 
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = prefix + "patch_embeds." + index_layer + ".norm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)

            # Convert segformer's layernorm layers
            elif "layer_norm_1" in key:
                index_block = key.split(".")[index_idx]
                index_block_ = key.split(".")[index_idx + 1]
                if "weight" in key:
                    new_key = prefix + "blocks." + index_block + "." + index_block_ + ".norm1.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = prefix + "blocks." + index_block + "." + index_block_ + ".norm1.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            elif "layer_norm_2" in key:
                index_block = key.split(".")[index_idx]
                index_block_ = key.split(".")[index_idx + 1]
                if "weight" in key:
                    new_key = prefix + "blocks." + index_block + "." + index_block_ + ".norm2.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = prefix + "blocks." + index_block + "." + index_block_ + ".norm2.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            # Convert segformer's attention layers
            elif "attention" in key:
                index_block = key.split(".")[index_idx]
                index_block_ = key.split(".")[index_idx + 1]
                if "self.sr" in key:
                    if (
                        prefix + "blocks." + index_block + "." + index_block_ + ".attn.sr.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = prefix + "blocks." + index_block + "." + index_block_ + ".attn.sr.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                    
                elif "self.layer_norm" in key:
                    if (
                        prefix + "blocks." + index_block + "." + index_block_ + ".attn.norm.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = prefix + "blocks." + index_block + "." + index_block_ + ".attn.norm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                    
                elif "self.query" in key:
                    if (
                        prefix + "blocks." + index_block + "." + index_block_ + ".attn.q.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    q_w = key
                    k_w = q_w.replace("query", "key")
                    v_w = q_w.replace("query", "value")
                    q_b = q_w.replace("weight", "bias")
                    k_b = k_w.replace("weight", "bias")
                    v_b = v_w.replace("weight", "bias")

                    kv_w = flow.cat(
                        (
                            oneflow_state_dict.pop(k_w),
                            oneflow_state_dict.pop(v_w),
                        ),
                        dim=0,
                    )
                    kv_b = flow.cat(
                        (
                            oneflow_state_dict.pop(k_b),
                            oneflow_state_dict.pop(v_b),
                        ),
                        dim=-1,
                    )

                    new_key_q_w = prefix + "blocks." + index_block + "." + index_block_ + ".attn.q.weight"
                    oneflow_state_dict[new_key_q_w] = oneflow_state_dict.pop(q_w)
                    new_key_kv_w = prefix + "blocks." + index_block + "." + index_block_ + ".attn.kv.weight"
                    oneflow_state_dict[new_key_kv_w] = kv_w

                    new_key_q_b = new_key_q_w.replace("weight", "bias")
                    oneflow_state_dict[new_key_q_b] = oneflow_state_dict.pop(q_b)
                    new_key_kv_b = new_key_kv_w.replace("weight", "bias")
                    oneflow_state_dict[new_key_kv_b] = kv_b
                
                elif "output" in key:
                    if "dense" in key:
                        if "weight" in key:
                            new_key = prefix + "blocks." + index_block + "." + index_block_ + ".attn.proj.weight"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        if "bias" in key:
                            new_key = prefix + "blocks." + index_block + "." + index_block_  + ".attn.proj.bias"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            elif "mlp" in key:
                index_block = key.split(".")[index_idx]
                index_block_ = key.split(".")[index_idx + 1]
                if "dense1.weight" in key:
                    if (
                        prefix + "blocks." + index_block + "." + index_block_ + ".mlp.dense_h_to_4h.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = prefix + "blocks." + index_block + "." + index_block_ + ".mlp.dense_h_to_4h.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                
                elif "dwconv.weight" in key:
                    if (
                        prefix + "blocks." + index_block + "." + index_block_ + ".mlp.dwconv.dwconv.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = prefix + "blocks." + index_block + "." + index_block_ + ".mlp.dwconv.dwconv.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                    
                elif "dense2.weight" in key:
                    if (
                        prefix + "blocks." + index_block + "." + index_block_ + ".mlp.dense_4h_to_h.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = prefix + "blocks." + index_block + "." + index_block_ + ".mlp.dense_4h_to_h.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                    
            elif "layer_norm" in key:
                index_block = key.split(".")[index_idx]
                if (
                    prefix + "layer_norms." + index_block + ".weight"
                    in oneflow_state_dict.keys()
                ):
                    continue
                w = key
                b = key.replace("weight", "bias")
                new_key = prefix + "layer_norms." + index_block + ".weight"
                oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                new_key = new_key.replace("weight", "bias")
                oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)

            elif "linear_c" in key:
                index_de = key.split(".")[index_decode]
                if "proj.weight" in key:
                    if (
                        "head.linear_c." + index_de + ".proj.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    new_key = "head.linear_c." + index_de + ".proj.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
            
            elif "linear_fuse" in key:
                new_key = "head.linear_fuse.weight"
                oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                
            elif "batch_norm" in key:
                name = key.split(".")[2]
                if name == "num_batches_tracked":
                    oneflow_state_dict.pop(key)
                else:
                    new_key = "head.batch_norm." + name
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
            
            elif "classifier" in key:
                if "weight" in key:
                    new_key = "head.linear_pred.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = "head.linear_pred.bias"
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

        # update libai_cfg by config.json
        self.libai_cfg.img_size = cfg_dict["image_size"]
        self.libai_cfg.patch_sizes = cfg_dict["patch_sizes"]
        self.libai_cfg.strides = cfg_dict["strides"]
        self.libai_cfg.in_chans = cfg_dict["num_channels"]
        self.libai_cfg.num_blocks = cfg_dict["num_encoder_blocks"]
        self.libai_cfg.embed_dims = cfg_dict["hidden_sizes"]
        self.libai_cfg.num_heads = cfg_dict["num_attention_heads"]
        self.libai_cfg.mlp_ratios = cfg_dict["mlp_ratios"]
        self.libai_cfg.drop_rate = cfg_dict["hidden_dropout_prob"]
        self.libai_cfg.attn_drop_rate = cfg_dict["attention_probs_dropout_prob"]
        self.libai_cfg.drop_path_rate = cfg_dict["drop_path_rate"]
        self.libai_cfg.decoder_dropout_prob = cfg_dict["classifier_dropout_prob"]
        self.libai_cfg.depths = cfg_dict["depths"]
        self.libai_cfg.sr_ratios = cfg_dict["sr_ratios"]
        self.libai_cfg.decoder_embedding_dim = cfg_dict["decoder_hidden_size"]

        # update libai_cfg by kwargs
        for k, v in self.kwargs.items():
            self.libai_cfg[k] = v
            
class SegFLoaderLiBai(ModelLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = ""