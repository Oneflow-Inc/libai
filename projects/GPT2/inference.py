from libai.models.utils.model_utils.gpt_loader import GPT2LoaderHuggerFace
from projects.GPT2.gpt2 import GPTModel
from projects.GPT2.configs.gpt_inference import cfg as inference_cfg
from libai.tokenizer import GPT2Tokenizer

path = "/path/to/gpt2"
text = ["studies have shown that owning a dog is good for you"]


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer(vocab_file=path + "vocab.json", merges_file=path + "merges.txt")
    input_ids = tokenizer.encode(text, return_tensors='of', is_global=True,)

    loader = GPT2LoaderHuggerFace(GPTModel, inference_cfg, path,)

    gpt2 = loader.load()
    gpt2.eval()
    
    outputs = gpt2.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # studies have shown that owning a dog is good for you, but it's not the only way