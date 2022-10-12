import os

import oneflow as flow
from omegaconf import DictConfig

from .model_utils.base_loader import WEIGHTS_NAME_PT
from .model_utils.bert_loader import BertLoaderHuggerFace, BertLoaderLiBai
from .model_utils.gpt_loader import GPT2LoaderHuggerFace, GPT2LoaderLiBai
from .model_utils.roberta_loader import RobertaLoaderHuggerFace, RobertaLoaderLiBai
from .model_utils.swin_loader import SwinLoaderHuggerFace, SwinLoaderLiBai
from .model_utils.swinv2_loader import SwinV2LoaderHuggerFace, SwinV2LoaderLiBai
from .model_utils.vit_loader import ViTLoaderHuggerFace, ViTLoaderLiBai

# from configs.common.models.bert import cfg as bert_cfg


# from configs.common.models.roberta import cfg as roberta_cfg
# from configs.common.models.gpt import cfg as gpt_cfg
# from configs.common.models.t5 import cfg as t5_cfg
# from configs.common.models.swin.swin_tiny_patch4_window7_224 import cfg as swin_cfg
# from configs.common.models.swinv2.swinv2_tiny_patch4_window8_256 import cfg as swinv2_cfg


class PretrainedModel(flow.nn.Module):
    model_loader_huggerface = None
    model_loader_libai = None
    default_cfg = None

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_pretrained(cls, pretrained_path, libai_cfg=None, **kwargs):
        assert cls.model_loader_libai is not None, "model_loader_libai cannot be None"
        assert cls.model_loader_huggerface is not None, "model_loader_huggerface cannot be None"

        libai_cfg = libai_cfg if libai_cfg is not None else cls.default_cfg

        if os.path.isfile(os.path.join(pretrained_path, WEIGHTS_NAME_PT)):
            loader = cls.model_loader_huggerface(cls, libai_cfg, pretrained_path, **kwargs)
        else:
            loader = cls.model_loader_libai(cls, libai_cfg, pretrained_path, **kwargs)

        return loader.load()


class PretrainedBert(PretrainedModel):
    model_loader_huggerface = BertLoaderHuggerFace
    model_loader_libai = BertLoaderLiBai
    default_cfg = dict(
        vocab_size=30522,
        hidden_size=768,
        hidden_layers=24,
        num_attention_heads=12,
        intermediate_size=4096,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        num_tokentypes=2,
        add_pooling_layer=True,
        initializer_range=0.02,
        layernorm_eps=1e-5,
        bias_gelu_fusion=True,
        bias_dropout_fusion=True,
        scale_mask_softmax_fusion=True,
        apply_query_key_layer_scaling=True,
        apply_residual_post_layernorm=False,
        add_binary_head=True,
        amp_enabled=False,
    )
    default_cfg = DictConfig(default_cfg)
    libai_cfg = default_cfg
