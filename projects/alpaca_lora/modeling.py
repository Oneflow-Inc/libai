from projects.mock_transformers import init_env  # noqa
# from oneflow.utils.global_view import global_mode
from typing import Optional
from transformers import AutoModelForSeq2SeqLM
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import oneflow as flow
from oneflow import nn
# import torch as flow


class AlpacaModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        base_model = "decapoda-research/llama-7b-hf"

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = LlamaForCausalLM.from_pretrained(
            base_model,
            flow_dtype=flow.float16,
            load_in_8bit=True,
            device_map="auto",
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        self.tokenizer.padding_side = "left"  # Allow batched inference
        model = prepare_model_for_int8_training(model)
        self.model = get_peft_model(model, peft_config)
        # model.print_trainable_parameters()
    
    def forward(
        self,
        input_ids: flow.LongTensor = None,
        attention_mask: Optional[flow.Tensor] = None,
        position_ids: Optional[flow.LongTensor] = None,
        past_key_values: Optional[List[flow.FloatTensor]] = None,
        inputs_embeds: Optional[flow.FloatTensor] = None,
        labels: Optional[flow.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    )

    ):
        return self.model(**argv)
