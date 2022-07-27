import json

import oneflow as flow

from .base_loader import ModelLoaderHuggerFace, ModelLoaderLiBai


class BertLoaderHuggerFace(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)

        """NOTE: base_model_prefix_1 is BERT's prefix in Transformers.
        base_model_prefix_2 is BERT's prefix in LiBai."""
        self.base_model_prefix_1 = "bert"
        self.base_model_prefix_2 = "bert"

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

        # Get configs
        num_heads = cfg.get("num_attention_heads")
        hidden_size = cfg.get("hidden_size")
        layers = cfg.get("hidden_layers")
        head_size = int(hidden_size / num_heads)

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)

        prefix = "bert." if has_prefix else ""
        index_idx = 3 if has_prefix else 2
        qkv_idx = 6 if has_prefix else 5

        old_keys = oneflow_state_dict.keys()

        for key in list(old_keys):

            # Convert bert's embedding layers
            if "embeddings" in key:
                if "word_embeddings" in key:
                    new_key = key.replace("word_embeddings", "vocab_embeddings")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "token_type_embeddings" in key:
                    new_key = key.replace("token_type_embeddings", "tokentype_embeddings")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "LayerNorm.weight" in key:
                    new_key = prefix + "encoders.0.input_layernorm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "LayerNorm.bias" in key:
                    new_key = prefix + "encoders.0.input_layernorm.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                else:
                    oneflow_state_dict[key] = oneflow_state_dict[key]

            # Convert bert's attention layers
            elif "attention" in key:
                if "self" in key:
                    index = key.split(".")[index_idx]
                    if (
                        prefix + "encoders." + index + ".self_attention.query_key_value.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    q_w = key.replace(key.split(".")[qkv_idx], "query").replace(
                        key.split(".")[qkv_idx + 1], "weight"
                    )
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

                    qkv_w = self._fix_qkv_ordering(qkv_w, head_size, num_heads)
                    qkv_b = self._fix_qkv_ordering(qkv_b, head_size, num_heads)

                    new_key = (
                        prefix + "encoders." + index + ".self_attention.query_key_value.weight"
                    )
                    oneflow_state_dict[new_key] = qkv_w

                    new_key = prefix + "encoders." + index + ".self_attention.query_key_value.bias"
                    oneflow_state_dict[new_key] = qkv_b
                elif "output" in key:
                    index = key.split(".")[index_idx]
                    if "dense" in key:
                        if "weight" in key:
                            new_key = prefix + "encoders." + index + ".self_attention.dense.weight"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        elif "bias" in key:
                            new_key = prefix + "encoders." + index + ".self_attention.dense.bias"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif "LayerNorm" in key:
                        if "weight" in key:
                            new_key = (
                                prefix + "encoders." + index + ".post_attention_layernorm.weight"
                            )
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        elif "bias" in key:
                            new_key = (
                                prefix + "encoders." + index + ".post_attention_layernorm.bias"
                            )
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            # Convert bert's intermediate layers
            elif "intermediate" in key:
                index = key.split(".")[index_idx]
                if (
                    prefix + "encoders." + index + ".mlp.dense_h_to_4h.weight"
                    in oneflow_state_dict.keys()
                ):
                    continue
                if "weight" in key:
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = prefix + "encoders." + index + ".mlp.dense_h_to_4h.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)

            # Convert bert's output layers
            elif "output" in key:
                index = key.split(".")[index_idx]
                if "dense.weight" in key:
                    if (
                        prefix + "encoders." + index + ".mlp.dense_4h_to_h.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    new_key = prefix + "encoders." + index + ".mlp.dense_4h_to_h.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                elif "LayerNorm.weight" in key:
                    if (
                        prefix + "encoders." + str(int(index) + 1) + ".input_layernorm.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    if index == str(layers - 1):
                        new_key = prefix + "final_layernorm.weight"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                        new_key = new_key.replace("weight", "bias")
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)
                        continue
                    new_key = prefix + "encoders." + str(int(index) + 1) + ".input_layernorm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(w)
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(b)

            # Convert bert's pooler layers
            elif "pooler" in key:
                if "weight" in key:
                    new_key = prefix + "pooler.dense.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "bias" in key:
                    new_key = prefix + "pooler.dense.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            # Convert cls_head layers
            elif "cls" in key:
                if "predictions.bias" in key:
                    new_key = "cls_head.lm_logits.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "dense.weight" in key:
                    new_key = "cls_head.predictions.dense.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "dense.bias" in key:
                    new_key = "cls_head.predictions.dense.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "LayerNorm.weight" in key:
                    new_key = "cls_head.predictions.layernorm.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "LayerNorm.bias" in key:
                    new_key = "cls_head.predictions.layernorm.bias"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif "seq_relationship" in key:
                    new_key = key.replace("cls", "cls_head")
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
        for k, v in cfg_dict.items():
            if k == "num_hidden_layers":
                self.libai_cfg.hidden_layers = v
            elif k == "type_vocab_size":
                self.libai_cfg.num_tokentypes = v
            elif k == "layer_norm_eps":
                self.libai_cfg.layernorm_eps = v
            elif k in cfg_dict:
                self.libai_cfg[k] = v

        # update libai_cfg by kwargs
        for k, v in self.kwargs.items():
            self.libai_cfg[k] = v

        # use original BERT residual connection ordering
        self.libai_cfg.apply_residual_post_layernorm = True


class BertLoaderLiBai(ModelLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = "bert"
