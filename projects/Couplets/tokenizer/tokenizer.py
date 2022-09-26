import collections


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            token = token.strip("\n")
            if not token:
                break
            vocab[token] = index
            index += 1
    return vocab


class CoupletsTokenizer:
    def __init__(self, vocab_file):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_tokens = self.vocab.keys()
        self.unk_id = self.vocab["<unk>"]
        self.pad_id = self.vocab["<pad>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]

    def tokenize(self, text):
        tokens_list = text.split()
        return tokens_list

    def convert_tokens_to_ids(self, tokens_list):
        ids_list = []
        for token in tokens_list:
            if token not in self.vocab_tokens:
                token = "<unk>"
            token_id = self.vocab[token]
            ids_list.append(token_id)
        return ids_list

    def convert_ids_to_tokens(self, ids_list):
        tokens_list = []
        for token_id in ids_list:
            token = self.inv_vocab[token_id]
            tokens_list.append(token)
        return tokens_list
