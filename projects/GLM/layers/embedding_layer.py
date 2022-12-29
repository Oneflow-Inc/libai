import oneflow as flow
from oneflow import nn
from projects.GLM.layers.position_embedding import SinePositionalEmbedding as PositionalEmbedding
from libai.layers import VocabEmbedding, Embedding
import libai.utils.distributed as dist
from libai.models.utils import init_method_normal

class GLMEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_seq_length,
        init_method=init_method_normal(0.02, 0),
        embedding_dropout_prob=0.0,
        amp_enabled=False,
        block_position_encoding=False,
    ):
        super().__init__()
        self.block_position_encoding = block_position_encoding
        
        self.word_embeddings = VocabEmbedding(
            vocab_size, hidden_size, init_method=init_method, amp_enabled=amp_enabled
        )
        if block_position_encoding:
            self.position_embeddings = Embedding(
                max_seq_length + 1, hidden_size, init_method=init_method, amp_enabled=amp_enabled
            )
            self.block_position_embeddings = Embedding(
                max_seq_length + 1, hidden_size, init_method=init_method, amp_enabled=amp_enabled
            )
        self.embedding_dropout = nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids, position_ids):        
        word_embeddings = self.word_embeddings(input_ids)
        
        if self.block_position_encoding:
            position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        
        position_embeddings = self.position_embeddings(position_ids)
        input_embeddings = word_embeddings + position_embeddings

        if self.block_position_encoding:
            block_position_embeddings = self.block_position_embeddings(block_position_ids)
            input_embeddings = input_embeddings + block_position_embeddings
        
        input_embeds = self.dropout(input_embeds)
        return input_embeds

