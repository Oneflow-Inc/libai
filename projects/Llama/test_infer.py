from projects.Llama.configs.llama_config import cfg, tokenizer
from projects.Llama.llama import LlamaForCausalLM
from projects.Llama.llama_loader import LlamaLoaderHuggerFace
from libai.config import instantiate


pretrained_model_path = "/data0/hf_models/meta-llama/Llama-2-7b-hf"

text = "a dog is flying on the sky"
tokenizer = instantiate(tokenizer)
input_ids = tokenizer.tokenize(text, add_bos=True)

load_func = LlamaLoaderHuggerFace(
    model=LlamaForCausalLM,
    libai_cfg=cfg,
    pretrained_model_path=pretrained_model_path,
)
model = load_func.load()
model.eval()

res = model.generate(
    input_ids,
    max_new_tokens=20,
    do_sample=False,
)

res = tokenizer.decode(res)
print(res)

