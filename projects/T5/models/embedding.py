import oneflow as flow
from libai.layers.embedding import VocabEmbedding


class T5Embedding(flow.nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method=flow.nn.init.xavier_normal_,
        amp_enabled=False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.word_embeddings = VocabEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            init_method=init_method,
            amp_enabled=amp_enabled,
        )

        self.embedding_dropout = flow.nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids):
        word_embeddings = self.word_embeddings(input_ids)
        embeddings = self.embedding_dropout(word_embeddings)
        return embeddings