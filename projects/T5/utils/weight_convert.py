import argparse

import oneflow as flow
import torch

from libai.config import LazyConfig


def parse_args():
    parser = argparse.ArgumentParser(description="MT5 Weight Convertor")
    parser.add_argument(
        "--oneflow_state_dict_path", type=str, help="The path of mt5's checkpoint in LiBai"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="projects/T5/configs/mt5_pretrain.py",
        help="The path of the training config",
    )
    parser.add_argument("--save_path", type=str, default="projects/T5/pytorch_model.bin")
    return parser.parse_args()


def fix_qkv_ordering(qkv, head_size, num_heads, hidden_size=None):
    hidden_size = (head_size * num_heads) if hidden_size is None else hidden_size
    num_of_qkv = qkv.shape[0] // (head_size * num_heads)

    qkv = qkv.view(-1)
    qkv = qkv.view(num_heads, num_of_qkv, head_size, hidden_size)
    qkv = qkv.permute(1, 0, 2, 3).contiguous()
    qkv = qkv.view(num_of_qkv * head_size * num_heads, hidden_size)
    return qkv


def convert_tensor(tensor):
    return torch.tensor(tensor.detach().to_numpy(), dtype=torch.float32)


def convert_state_dict(oneflow_state_dict_path, libai_cfg, prefix="t5_model."):
    oneflow_state_dict = flow.load(oneflow_state_dict_path)
    torch_state_dict = {}
    # Get configs
    num_heads = libai_cfg.get("num_attention_heads")
    hidden_size = libai_cfg.get("hidden_size")
    head_size = libai_cfg.get("head_size", None)

    if head_size is None:
        head_size = int(hidden_size / num_heads)

    layer_idx = 3 if len(prefix) > 1 else 2
    enc_dec_idx = 1 if len(prefix) > 1 else 0
    op_idx = 4 if len(prefix) > 1 else 3

    # Convert T5's Embedding layers.
    x = convert_tensor(oneflow_state_dict.pop(prefix + "embedding.word_embeddings.weight"))
    new_key = "shared.weight"
    torch_state_dict[new_key] = x
    new_key = "encoder.embed_tokens.weight"
    torch_state_dict[new_key] = x
    new_key = "decoder.embed_tokens.weight"
    torch_state_dict[new_key] = x

    # Convert T5's final_layer_norm
    new_key = "encoder.final_layer_norm.weight"
    torch_state_dict[new_key] = convert_tensor(
        oneflow_state_dict.pop(prefix + "encoder.final_layernorm.weight")
    )
    new_key = "decoder.final_layer_norm.weight"
    torch_state_dict[new_key] = convert_tensor(
        oneflow_state_dict.pop(prefix + "decoder.final_layernorm.weight")
    )

    old_keys = list(oneflow_state_dict.keys())

    # Convert T5's lm_head
    new_key = "lm_head.weight"
    if prefix + new_key in old_keys:
        torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(prefix + new_key))

    for key in old_keys:
        keys = key.split(".")
        if op_idx > len(keys):
            continue
        layers = keys[layer_idx]
        enc_dec = keys[enc_dec_idx]
        op_name = keys[op_idx]

        if keys[op_idx + 1] == "relative_attention_bias":
            new_key = enc_dec + ".block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))

        # Convert T5's Encoder layers.
        if enc_dec == "encoder":
            if op_name == "self_attention":
                if keys[op_idx + 1] == "query_key_value":
                    x = oneflow_state_dict.pop(key)
                    x = fix_qkv_ordering(x, head_size, num_heads, hidden_size)
                    q, k, v = flow.chunk(x, chunks=3, dim=0)

                    new_key = "encoder.block." + layers + ".layer.0.SelfAttention.q.weight"
                    torch_state_dict[new_key] = convert_tensor(q)
                    new_key = new_key.replace(".q", ".k")
                    torch_state_dict[new_key] = convert_tensor(k)
                    new_key = new_key.replace(".k", ".v")
                    torch_state_dict[new_key] = convert_tensor(v)
                if keys[op_idx + 1] == "dense":
                    new_key = "encoder.block." + layers + ".layer.0.SelfAttention.o.weight"
                    torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))

            elif op_name == "input_layernorm":
                new_key = "encoder.block." + layers + ".layer.0.layer_norm.weight"
                torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))

            elif op_name == "post_attention_layernorm":
                new_key = "encoder.block." + layers + ".layer.1.layer_norm.weight"
                torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))

            elif op_name == "mlp":
                if libai_cfg.get("model_type") == "mt5":
                    if keys[op_idx + 1] == "wi_0":
                        new_key = "encoder.block." + layers + ".layer.1.DenseReluDense.wi_0.weight"
                        torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))
                    if keys[op_idx + 1] == "wi_1":
                        new_key = "encoder.block." + layers + ".layer.1.DenseReluDense.wi_1.weight"
                        torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))
                    if keys[op_idx + 1] == "wo":
                        new_key = "encoder.block." + layers + ".layer.1.DenseReluDense.wo.weight"
                        torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))
                elif libai_cfg.get("model_type") == "t5":
                    if keys[op_idx + 1] == "dense_h_to_4h":
                        new_key = "encoder.block." + layers + ".layer.1.DenseReluDense.wi.weight"
                        torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))
                    if keys[op_idx + 1] == "dense_4h_to_h":
                        new_key = "encoder.block." + layers + ".layer.1.DenseReluDense.wo.weight"
                        torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))

        # Convert T5's decoder Layers.
        elif enc_dec == "decoder":
            if op_name == "self_attention":
                if keys[op_idx + 1] == "query_key_value":
                    x = oneflow_state_dict.pop(key)
                    x = fix_qkv_ordering(x, head_size, num_heads, hidden_size)
                    q, k, v = flow.chunk(x, chunks=3, dim=0)

                    new_key = "decoder.block." + layers + ".layer.0.SelfAttention.q.weight"
                    torch_state_dict[new_key] = convert_tensor(q)
                    new_key = new_key.replace(".q", ".k")
                    torch_state_dict[new_key] = convert_tensor(k)
                    new_key = new_key.replace(".k", ".v")
                    torch_state_dict[new_key] = convert_tensor(v)
                if keys[op_idx + 1] == "dense":
                    new_key = "decoder.block." + layers + ".layer.0.SelfAttention.o.weight"
                    torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))

            elif op_name == "input_layernorm":
                new_key = "decoder.block." + layers + ".layer.0.layer_norm.weight"
                torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))

            elif op_name == "post_attention_layernorm":
                new_key = "decoder.block." + layers + ".layer.1.layer_norm.weight"
                torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))

            elif op_name == "post_cross_attention_layernorm":
                new_key = "decoder.block." + layers + ".layer.2.layer_norm.weight"
                torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))

            elif op_name == "cross_attention":
                if keys[op_idx + 1] == "query":
                    x = oneflow_state_dict.pop(key)
                    x = fix_qkv_ordering(x, head_size, num_heads, hidden_size)
                    new_key = "decoder.block." + layers + ".layer.1.EncDecAttention.q.weight"
                    torch_state_dict[new_key] = convert_tensor(x)
                if keys[op_idx + 1] == "key_value":
                    x = oneflow_state_dict.pop(key)
                    x = fix_qkv_ordering(x, head_size, num_heads, hidden_size)
                    k, v = flow.chunk(x, chunks=2, dim=0)
                    new_key = "decoder.block." + layers + ".layer.1.EncDecAttention.k.weight"
                    torch_state_dict[new_key] = convert_tensor(k)
                    new_key = new_key.replace(".k", ".v")
                    torch_state_dict[new_key] = convert_tensor(v)
                if keys[op_idx + 1] == "dense":
                    new_key = "decoder.block." + layers + ".layer.1.EncDecAttention.o.weight"
                    torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))

            elif op_name == "mlp":
                if libai_cfg.get("model_type") == "mt5":
                    if keys[op_idx + 1] == "wi_0":
                        new_key = "decoder.block." + layers + ".layer.2.DenseReluDense.wi_0.weight"
                        torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))
                    if keys[op_idx + 1] == "wi_1":
                        new_key = "decoder.block." + layers + ".layer.2.DenseReluDense.wi_1.weight"
                        torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))
                    if keys[op_idx + 1] == "wo":
                        new_key = "decoder.block." + layers + ".layer.2.DenseReluDense.wo.weight"
                        torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))
                elif libai_cfg.get("model_type") == "t5":
                    if keys[op_idx + 1] == "dense_h_to_4h":
                        new_key = "decoder.block." + layers + ".layer.2.DenseReluDense.wi.weight"
                        torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))
                    if keys[op_idx + 1] == "dense_4h_to_h":
                        new_key = "decoder.block." + layers + ".layer.2.DenseReluDense.wo.weight"
                        torch_state_dict[new_key] = convert_tensor(oneflow_state_dict.pop(key))

    return torch_state_dict


if __name__ == "__main__":
    args = parse_args()
    oneflow_state_dict_path = args.oneflow_state_dict_path
    config_path = args.config_path
    save_path = args.save_path

    training_config = LazyConfig.load(config_path)
    model_config = training_config.model.cfg
    torch_state_dict = convert_state_dict(oneflow_state_dict_path, model_config)

    torch.save(torch_state_dict, save_path)
