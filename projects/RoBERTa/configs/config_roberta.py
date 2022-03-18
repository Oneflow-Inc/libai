from libai.config import LazyCall

from ..roberta_model import RoBERTaModel, RoBERTaForPreTraining

cfg = dict(
    vocab_size=49624,
    max_position_embeddings=512,
    type_vocab_size=2,
    hidden_size=36,  # 768
    hidden_dropout_prob=0.1,
    pad_token_id=1,
    position_embedding_type="absolute",
    num_layers=2, # 12
    intermediate_size=3072,
    nheads=12,
    activation="gelu",
    chunk_size_feed_forward=0,
    layer_norm_eps=0.00001,
    attn_dropout=0,
    is_decoder=False,
    add_cross_attention=False,
    add_pooling_layer=True,
    initializer_range=0.02,
)

bert_model = LazyCall(RoBERTaModel)(cfg=cfg)

pretrain_model = LazyCall(RoBERTaForPreTraining)(cfg=cfg)
