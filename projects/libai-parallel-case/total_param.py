from transformers import GPT2LMHeadModel, GPT2Config

config = GPT2Config.from_json_file("/home/xiezipeng/workspace/libai/projects/libai-parallel-case/gpt2/gpt2-xl.json")
model = GPT2LMHeadModel(config)
print(model)


"""
wte: 50257 * 1600 = 80411200
wpe: 1024 * 1600 = 1638400
ln_f: 1600
lm_head: 1600 * 50257 = 80411200
h: {
    ln_1: 1600
    attn: {
        c_attn: 3 * 1600 * 1600 = 7680000
        c_proj: 1600 * 1600 = 2560000
    }
    ln_2: 1600
    mlp: {
        c_fc: 4 * 1600 * 1600 = 10240000
        c_proj: 1600 * 4 * 1600 = 10240000
    }
} = 30723200

80411200 + 1638400 + 1600 + 80411200 + 30723200 * 48 = 1637176000
"""