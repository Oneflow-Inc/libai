from libai.config import LazyCall as L
from ..modeling.model import Roberta
import oneflow as flow
from libai.optim import get_default_optimizer_params, PolynomialLR


# 模型中的参数
roberta_cfg = dict(
    vocab_size=49624,
    max_position_embeddings=512,
    type_vocab_size=2,
    hidden_size=768,
    hidden_dropout=0,
    pad_token_id=1,
    position_embedding_type="absolute",
    num_layers=12,
    intermediate_size=3072,
    nheads=12,
    activation="gelu",
    chunk_size_feed_forward=0,
    layer_norm_eps=0.00001,
    attn_dropout=0,
    is_decoder=False,
    add_cross_attention=False
)

model = L(Roberta)(cfg=roberta_cfg)

# 设置optim的方法，但是没看懂，大致根据我的理解写了一下
optim = L(flow.optim.Adam)(
    parameters=L(get_default_optimizer_params)(
        # 这里面的参数是做什么用的？
    ),
    lr=0.000001
)

# graph是干什么用的，是什么意思？