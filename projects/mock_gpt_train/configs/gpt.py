from projects.mock_transformers import init_env  # noqa
from projects.mock_transformers.dist_infer_gpt import *
from libai.config import LazyCall
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config

cfg = LazyCall(GPT2Config)(vocab_size=50257)

gpt_model = LazyCall(GPT2Model)(config=cfg)

pretrain_model = LazyCall(GPT2LMHeadModel)(config=cfg)
