from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/data0/hf_models/meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("/data0/hf_models/meta-llama/Llama-2-7b-hf").to("cuda")

text = "a dog is flying on the sky"

inputs = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

res = model.generate(
    inputs,
    max_new_tokens=20,
    do_sample=False,  # greedy search
)

res = tokenizer.decode(res[0])
print(repr(res))
