from numpy import dtype, mod
from numpy.lib.function_base import place
import megatron
import oneflow as flow
from libai.utils import distributed as dist
from libai.layers import (
    Embedding,
    VocabEmbedding,
    TransformerLayer,
)
from oneflow import sbp



class T5Embedding(flow.nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method=flow.nn.init.xavier_normal_,
        num_tokentypes=0,
        fp16=False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_tokentypes = num_tokentypes

        self.word_embeddings = VocabEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            init_method=init_method,
            fp16=fp16,
        )
        self.position_embeddings = Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=hidden_size,
            init_method=init_method,
            fp16=fp16,
        )
        self.position_ids = flow.arange(
            max_sequence_length,
            dtype=flow.long,
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
            placement=dist.get_layer_placement(0),
        ).unsqueeze(0)

        if self.num_tokentypes > 0:
            self.tokentype_embedding = Embedding(
                num_embeddings=self.num_tokentypes,
                embedding_dim=self.hidden_size,
                init_method=init_method,
                fp16=fp16,
            )
            self.tokentype_ids = flow.zeros(
                self.position_ids.size(),
                dtype=flow.long,
                sbp=self.position_ids.sbp,
                placement=self.position_ids.placement,
            )
        else:
            self.tokentype_embedding = None
            self.tokentype_ids = None

        self.embedding_dropout = flow.nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids, tokentype_ids=None, position_ids=None):
        # input_ids shape: [batch_size, seq_len, hidden_size]
        seq_length = input_ids.size()[1]
        word_embeddings = self.word_embeddings(input_ids)

        if position_ids is None:
            # Change position_ids sbp sign: [B, B] -> [S(0), B]
            position_ids = (
                self.position_ids[:, :seq_length]
                .expand_as(input_ids)
                .to_consistent(sbp=input_ids.sbp)
            )
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings

        if self.tokentype_embedding is not None:
            if tokentype_ids is None:
                tokentype_ids = (
                    self.tokentype_ids[:, :seq_length]
                    .expand_as(input_ids)
                    .to_consistent(sbp=input_ids.sbp)
                )
            embeddings = embeddings + self.tokentype_embedding(tokentype_ids)
        embeddings = self.embedding_dropout(embeddings)
        return embeddings
        
