import json

from .base_loader import ModelLoaderHuggerFace, ModelLoaderLiBai


class GPT2LoaderHuggerFace(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)

        """NOTE: base_model_prefix_1 is GPT's prefix in Transformers.
        base_model_prefix_2 is GPT's prefix in LiBai."""
        self.base_model_prefix_1 = "transformer"
        self.base_model_prefix_2 = "GPT_model"

    def _convert_state_dict(self, flow_state_dict, cfg):
        """Convert state_dict's keys to match model.

        Args:
            flow_state_dict (OrderedDict): model state dict.
            cfg (dict): model's default config dict in LiBai.

        Returns:
            OrderedDict: flow state dict.
        """
        # The converted checkpoint.
        oneflow_state_dict = flow_state_dict.copy()
        old_keys = list(oneflow_state_dict.keys())

        # Get configs
        num_heads = cfg.get("num_attention_heads")
        hidden_size = cfg.get("hidden_size")
        head_size = int(hidden_size / num_heads)

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)
        prefix1 = self.base_model_prefix_1 + "." if has_prefix else ""
        prefix2 = "GPT_model." if has_prefix else "GPT_model.transformer."
        layer_idx = 2 if has_prefix else 1

        # Convert Embedding layers.
        new_key = "GPT_model.embeddings.token_embeddings.weight"
        old_keys.remove(prefix1 + "wte.weight")
        oneflow_state_dict[new_key] = oneflow_state_dict.pop(prefix1 + "wte.weight")

        new_key = "GPT_model.embeddings.position_embeddings.weight"
        old_keys.remove(prefix1 + "wpe.weight")
        oneflow_state_dict[new_key] = oneflow_state_dict.pop(prefix1 + "wpe.weight")

        for key in old_keys:
            keys = key.split(".")
            if layer_idx > len(keys):
                continue
            layer = keys[layer_idx]
            # Convert transformer layers.
            if "h." in key:
                if "ln_1" in key:
                    if "weight" in key:
                        new_key = prefix2 + "layers." + layer + ".input_layernorm.weight"
                    else:
                        new_key = prefix2 + "layers." + layer + ".input_layernorm.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "ln_2" in key:
                    if "weight" in key:
                        new_key = prefix2 + "layers." + layer + ".post_attention_layernorm.weight"
                    else:
                        new_key = prefix2 + "layers." + layer + ".post_attention_layernorm.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "attn" in key:
                    if "c_attn" in key:
                        if "weight" in key:
                            new_key = (
                                prefix2
                                + "layers."
                                + layer
                                + ".self_attention.query_key_value.weight"
                            )
                        else:
                            new_key = (
                                prefix2 + "layers." + layer + ".self_attention.query_key_value.bias"
                            )
                        qkv = oneflow_state_dict.pop(key)
                        if qkv.ndim > 1:
                            qkv = qkv.transpose(1, 0)
                        qkv = self._fix_qkv_ordering(qkv, head_size, num_heads)
                        oneflow_state_dict[new_key] = qkv
                    elif "c_proj" in key:
                        if "weight" in key:
                            new_key = prefix2 + "layers." + layer + ".self_attention.dense.weight"
                        elif "bias" in key:
                            new_key = prefix2 + "layers." + layer + ".self_attention.dense.bias"
                        value = oneflow_state_dict.pop(key)
                        if value.ndim > 1:
                            value = value.transpose(1, 0)
                        oneflow_state_dict[new_key] = value
                elif "mlp" in key:
                    if "c_fc" in key:
                        if "weight" in key:
                            new_key = prefix2 + "layers." + layer + ".mlp.dense_h_to_4h.weight"
                        elif "bias" in key:
                            new_key = prefix2 + "layers." + layer + ".mlp.dense_h_to_4h.bias"
                        value = oneflow_state_dict.pop(key)
                        if value.ndim > 1:
                            value = value.transpose(1, 0)
                        oneflow_state_dict[new_key] = value
                    elif "c_proj" in key:
                        if "weight" in key:
                            new_key = prefix2 + "layers." + layer + ".mlp.dense_4h_to_h.weight"
                        elif "bias" in key:
                            new_key = prefix2 + "layers." + layer + ".mlp.dense_4h_to_h.bias"
                        value = oneflow_state_dict.pop(key)
                        if value.ndim > 1:
                            value = value.transpose(1, 0)
                        oneflow_state_dict[new_key] = value
            elif "ln_f" in key:
                if "weight" in key:
                    new_key = prefix2 + "layernorm_f.weight"
                elif "bias" in key:
                    new_key = prefix2 + "layernorm_f.bias"
                oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
        return oneflow_state_dict

    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.

        Args:
            config_file (str): Path of config file.
        """
        with open(config_file, mode="r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # update libai_cfg by config.json
        self.libai_cfg.num_layers = cfg_dict["n_layer"]
        self.libai_cfg.hidden_size = cfg_dict["n_embd"]
        self.libai_cfg.num_attention_heads = cfg_dict["n_head"]
        self.libai_cfg.max_seq_length = cfg_dict["n_positions"]
        self.libai_cfg.embedding_dropout_prob = cfg_dict["embd_pdrop"]
        self.libai_cfg.attention_dropout_prob = cfg_dict["attn_pdrop"]
        self.libai_cfg.output_dropout_prob = cfg_dict["resid_pdrop"]
        self.libai_cfg.layernorm_epsilon = cfg_dict["layer_norm_epsilon"]
        self.libai_cfg.vocab_size = cfg_dict["vocab_size"]
        self.libai_cfg.initializer_range = cfg_dict["initializer_range"]
        self.libai_cfg.ffn_hidden_size = cfg_dict.get("n_inner", 4 * self.libai_cfg["hidden_size"])

        # update libai_cfg by kwargs
        for k, v in self.kwargs.items():
            self.libai_cfg[k] = v


class GPT2LoaderLiBai(ModelLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = "GPT_model"
