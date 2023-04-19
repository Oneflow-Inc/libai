from projects.mock_transformers import init_env  # noqa
# from oneflow.utils.global_view import global_mode
from typing import Optional, List
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
    def __init__(
        self,
        model_name: str = "decapoda-research/llama-7b-hf",
        torch_dtype: Optional[flow.dtype] = flow.float16,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_task_type: str = "CAUSAL_LM",
    ):
        super().__init__()
        peft_config = LoraConfig(
            r = lora_r,
            lora_alpha = lora_alpha,
            target_modules = lora_target_modules,
            lora_dropout = lora_dropout,
            bias = lora_bias,
            task_type = lora_task_type,
        )

        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=flow.float16,
            # load_in_8bit=True,
            # device_map="auto",
        ).to("cuda")
        model = prepare_model_for_int8_training(model)
        self.model = get_peft_model(model, peft_config).to("cuda")
        # reset the state_dict to the new one
        old_state_dict = self.model.state_dict
        self.model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(self.model, type(self.model))
        self.state_dict = self.model.state_dict
    
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
    ):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return {
            "loss": output.loss,
            "logits": output.logits,
        }