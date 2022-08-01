import json

import oneflow as flow

from .base_loader import ModelLoaderHuggerFace, ModelLoaderLiBai


class T5LoaderHuggerFace(ModelLoaderHuggerFace):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)

        """NOTE: base_model_prefix_1 is T5's prefix in Transformers.
        base_model_prefix_2 is T5's prefix in LiBai."""
        self.base_model_prefix_1 = "transformer"
        self.base_model_prefix_2 = "t5_model"

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
        head_size = cfg.get("head_size", None)
        if head_size is None:
            head_size = int(hidden_size / num_heads)

        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)
        prefix1 = self.base_model_prefix_1 + "." if has_prefix else ""
        prefix2 = self.base_model_prefix_2 + "." if has_prefix else ""
        encoder_decoder_idx = 1 if has_prefix else 0
        layer_idx1 = 3 if has_prefix else 2
        layer_idx2 = 5 if has_prefix else 4
        op_idx = 6 if has_prefix else 5

        # Convert T5's Embedding layers.
        # NOTE: Transformers' T5 has no position embedding layer.
        new_key = prefix2 + "embedding.word_embeddings.weight"
        old_keys.remove(prefix1 + "shared.weight")
        oneflow_state_dict[new_key] = oneflow_state_dict.pop(prefix1 + "shared.weight")

        # Convert T5's final_layer_norm
        new_key = prefix2 + "encoder.final_layernorm.weight"
        old_keys.remove(prefix1 + "encoder.final_layer_norm.weight")
        oneflow_state_dict[new_key] = oneflow_state_dict.pop(
            prefix1 + "encoder.final_layer_norm.weight"
        )
        new_key = prefix2 + "decoder.final_layernorm.weight"
        old_keys.remove(prefix1 + "decoder.final_layer_norm.weight")
        oneflow_state_dict[new_key] = oneflow_state_dict.pop(
            prefix1 + "decoder.final_layer_norm.weight"
        )

        # NOTE: Each layers has no bias in Transformer's T5.
        for key in old_keys:
            keys = key.split(".")
            if layer_idx1 > len(keys) or layer_idx2 > len(keys):
                continue
            layer1 = keys[layer_idx1]
            layer2 = keys[layer_idx2]
            op_name = keys[op_idx]

            if keys[op_idx + 1] == "relative_attention_bias" and keys[op_idx] == "SelfAttention":
                new_key = (
                    prefix2
                    + keys[encoder_decoder_idx]
                    + ".layers.0.self_attention.relative_attention_bias.weight"
                )
                oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            # Convert T5's Encoder layers.
            if keys[encoder_decoder_idx] == "encoder":
                if op_name == "SelfAttention":
                    new_key = (
                        prefix2
                        + "encoder.layers."
                        + layer1
                        + ".self_attention.query_key_value.weight"
                    )
                    if new_key in oneflow_state_dict.keys():
                        continue
                    q_w = ".".join(keys[: op_idx + 1]) + ".q." + "weight"
                    k_w = ".".join(keys[: op_idx + 1]) + ".k." + "weight"
                    v_w = ".".join(keys[: op_idx + 1]) + ".v." + "weight"
                    qkv_w = flow.cat(
                        (
                            oneflow_state_dict.pop(q_w),
                            oneflow_state_dict.pop(k_w),
                            oneflow_state_dict.pop(v_w),
                        ),
                        dim=0,
                    )
                    qkv_w = self._fix_qkv_ordering(qkv_w, head_size, num_heads, hidden_size)
                    oneflow_state_dict[new_key] = qkv_w

                    o_w = ".".join(keys[: op_idx + 1]) + ".o." + "weight"
                    new_key = prefix2 + "encoder.layers." + layer1 + ".self_attention.dense.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(o_w)
                elif op_name == "layer_norm":
                    if layer2 == "0":
                        new_key = prefix2 + "encoder.layers." + layer1 + ".input_layernorm.weight"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif layer2 == "1":
                        new_key = (
                            prefix2
                            + "encoder.layers."
                            + layer1
                            + ".post_attention_layernorm.weight"
                        )
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif op_name == "DenseReluDense":
                    if cfg.get("mlp_type") == "t5":
                        if keys[op_idx + 1] == "wi":
                            new_key = (
                                prefix2 + "encoder.layers." + layer1 + ".mlp.dense_h_to_4h.weight"
                            )
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        elif keys[op_idx + 1] == "wo":
                            new_key = (
                                prefix2 + "encoder.layers." + layer1 + ".mlp.dense_4h_to_h.weight"
                            )
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif cfg.get("mlp_type") == "mt5":
                        if keys[op_idx + 1] == "wi_0":
                            new_key = prefix2 + "encoder.layers." + layer1 + ".mlp.wi_0.weight"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        elif keys[op_idx + 1] == "wi_1":
                            new_key = prefix2 + "encoder.layers." + layer1 + ".mlp.wi_1.weight"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        elif keys[op_idx + 1] == "wo":
                            new_key = prefix2 + "encoder.layers." + layer1 + ".mlp.wo.weight"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)

            # Convert T5's decoder Layers.
            elif keys[encoder_decoder_idx] == "decoder":
                if op_name == "SelfAttention":
                    new_key = (
                        prefix2
                        + "decoder.layers."
                        + layer1
                        + ".self_attention.query_key_value.weight"
                    )
                    if new_key in oneflow_state_dict.keys():
                        continue
                    q_w = ".".join(keys[: op_idx + 1]) + ".q." + "weight"
                    k_w = ".".join(keys[: op_idx + 1]) + ".k." + "weight"
                    v_w = ".".join(keys[: op_idx + 1]) + ".v." + "weight"
                    qkv_w = flow.cat(
                        (
                            oneflow_state_dict.pop(q_w),
                            oneflow_state_dict.pop(k_w),
                            oneflow_state_dict.pop(v_w),
                        ),
                        dim=0,
                    )
                    qkv_w = self._fix_qkv_ordering(qkv_w, head_size, num_heads, hidden_size)

                    oneflow_state_dict[new_key] = qkv_w

                    o_w = ".".join(keys[: op_idx + 1]) + ".o." + "weight"
                    new_key = prefix2 + "decoder.layers." + layer1 + ".self_attention.dense.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(o_w)
                elif op_name == "layer_norm":
                    if layer2 == "0":
                        new_key = prefix2 + "decoder.layers." + layer1 + ".input_layernorm.weight"
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif layer2 == "1":
                        new_key = (
                            prefix2
                            + "decoder.layers."
                            + layer1
                            + ".post_attention_layernorm.weight"
                        )
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif layer2 == "2":
                        new_key = (
                            prefix2
                            + "decoder.layers."
                            + layer1
                            + ".post_cross_attention_layernorm.weight"
                        )
                        oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                elif op_name == "EncDecAttention":
                    new_key = prefix2 + "decoder.layers." + layer1 + ".cross_attention.query.weight"
                    if new_key in oneflow_state_dict.keys():
                        continue
                    q_w = ".".join(keys[: op_idx + 1]) + ".q." + "weight"
                    k_w = ".".join(keys[: op_idx + 1]) + ".k." + "weight"
                    v_w = ".".join(keys[: op_idx + 1]) + ".v." + "weight"

                    q_w = oneflow_state_dict.pop(q_w)
                    kv_w = flow.cat(
                        (
                            oneflow_state_dict.pop(k_w),
                            oneflow_state_dict.pop(v_w),
                        ),
                        dim=0,
                    )
                    q_w = self._fix_qkv_ordering(q_w, head_size, num_heads, hidden_size)
                    kv_w = self._fix_qkv_ordering(kv_w, head_size, num_heads, hidden_size)

                    oneflow_state_dict[new_key] = q_w
                    new_key = (
                        prefix2 + "decoder.layers." + layer1 + ".cross_attention.key_value.weight"
                    )
                    oneflow_state_dict[new_key] = kv_w

                    o_w = ".".join(keys[: op_idx + 1]) + ".o." + "weight"
                    new_key = prefix2 + "decoder.layers." + layer1 + ".cross_attention.dense.weight"
                    oneflow_state_dict[new_key] = oneflow_state_dict.pop(o_w)
                elif op_name == "DenseReluDense":
                    if cfg.get("mlp_type") == "t5":
                        if keys[op_idx + 1] == "wi":
                            new_key = (
                                prefix2 + "decoder.layers." + layer1 + ".mlp.dense_h_to_4h.weight"
                            )
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        elif keys[op_idx + 1] == "wo":
                            new_key = (
                                prefix2 + "decoder.layers." + layer1 + ".mlp.dense_4h_to_h.weight"
                            )
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                    elif cfg.get("mlp_type") == "mt5":
                        if keys[op_idx + 1] == "wi_0":
                            new_key = prefix2 + "decoder.layers." + layer1 + ".mlp.wi_0.weight"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        elif keys[op_idx + 1] == "wi_1":
                            new_key = prefix2 + "decoder.layers." + layer1 + ".mlp.wi_1.weight"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
                        elif keys[op_idx + 1] == "wo":
                            new_key = prefix2 + "decoder.layers." + layer1 + ".mlp.wo.weight"
                            oneflow_state_dict[new_key] = oneflow_state_dict.pop(key)
        return oneflow_state_dict

    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.

        Args:
            config_file (str): Path of config file.
        """
        with open(config_file, mode="r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        self.libai_cfg.vocab_size = cfg_dict["vocab_size"]
        self.libai_cfg.hidden_size = cfg_dict["d_model"]
        self.libai_cfg.hidden_layers = cfg_dict["num_layers"]
        self.libai_cfg.num_attention_heads = cfg_dict["num_heads"]
        self.libai_cfg.intermediate_size = cfg_dict["d_ff"]
        self.libai_cfg.hidden_dropout_prob = cfg_dict["dropout_rate"]
        self.libai_cfg.attention_probs_dropout_prob = cfg_dict["dropout_rate"]
        self.libai_cfg.max_position_embeddings = cfg_dict.get("n_positions", 512)
        self.libai_cfg.relative_attention_num_buckets = cfg_dict["relative_attention_num_buckets"]
        self.libai_cfg.embedding_dropout_prob = cfg_dict["dropout_rate"]
        self.libai_cfg.initializer_range = cfg_dict["initializer_factor"]
        self.libai_cfg.layernorm_eps = cfg_dict["layer_norm_epsilon"]
        self.libai_cfg.head_size = cfg_dict["d_kv"]

        # update libai_cfg by kwargs
        for k, v in self.kwargs.items():
            self.libai_cfg[k] = v


class T5LoaderLibai(ModelLoaderLiBai):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = "t5_model"
