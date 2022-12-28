import oneflow as flow
from oneflow import nn

import libai.utils.distributed as dist
from libai.layers import MLP, VocabEmbedding
from libai.models.utils import init_method_normal, scaled_init_method_normal
from projects.GLM.layers.attention_layer import MultiheadAttention
from projects.GLM.layers.position_embedding import SinePositionalEmbedding as PositionalEmbedding
