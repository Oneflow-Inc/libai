from projects.mock_transformers import init_env  # noqa
from oneflow.utils.global_view import global_mode
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


base_model = "decapoda-research/llama-7b-hf"
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
    torch_dtype=flow.float16,
    # load_in_8bit=True,
    # device_map=None,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()