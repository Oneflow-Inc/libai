# bert-base-uncased
class config:
    def __init__(self):
        self.epochs = 1
        self.lr = 3e-5
        self.batch_size = 64
        self.dropout = 0.3
        self.pooler_type = 'cls'
        self.temp = 0.05
        self.train_data_path = '/home/xiezipeng/libai/projects/SimCSE/dataset/wiki1m_for_simcse.txt'
        self.dev_data_path = '/home/xiezipeng/libai/projects/SimCSE/dataset/sts_dev.txt'
        self.test_data_path = '/home/xiezipeng/libai/projects/SimCSE/dataset/sts_test.txt'
        self.vocab_path = '/home/xiezipeng/libai/projects/SimCSE/dataset/vocab.txt'
        self.save_path = '/home/xiezipeng/libai/projects/SimCSE/dataset/'
        self.vocab_size = 30522
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob=0.1
        self.max_position_embeddings = 512
        self.hidden_act = 'glue'
        self.layer_norm_eps = 1e-12
