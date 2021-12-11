from libai.config import LazyCall as L
from libai.models.gpt import GPT
from .bert import model

del model.add_pooler

model.blocks = L(GPT.make_default_blocks)()
