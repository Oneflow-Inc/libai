# from projects.mock_transformers import init_env  # noqa
import oneflow as flow
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from projects.alpaca_lora.data_utils import Prompter, generate_and_tokenize_prompt

from libai.data.structures import DistTensorData, Instance
from oneflow.utils.data import Dataset 

class LoraDataset(Dataset):
    def __init__(
        self,
        data_path: str = "yahma/alpaca-cleaned",
        prompt_dir: str = "data_test/lora_data/templates",
        prompt_template_name: str = "alpaca",
        tokenizer_name: str = "decapoda-research/llama-7b-hf",
        max_length: int = 256,
    ):
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference

        prompter = Prompter(prompt_dir, prompt_template_name)
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)

        self.train_data = data["train"].shuffle().map(
            lambda x: generate_and_tokenize_prompt(
                tokenizer, 
                prompter, 
                max_length, 
                x,
            )
        )

    def __getitem__(self, index):
        return Instance(
                input_ids=DistTensorData(flow.tensor(self.train_data[index]["input_ids"])),
                attention_mask=DistTensorData(flow.tensor(self.train_data[index]["attention_mask"])),
                labels=DistTensorData(flow.tensor(self.train_data[index]["labels"])
            )
        )

    def __len__(self):
        return len(self.train_data)
