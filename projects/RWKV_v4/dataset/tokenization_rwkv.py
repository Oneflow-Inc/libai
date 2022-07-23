from libai.tokenizer import GPT2Tokenizer
class RWKVTOKENIZER(GPT2Tokenizer):
    def __init__(        
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        **kwargs,):
        super().__init__()
        self.stoi = {v: int(k) for k, v in self.word_table.items()}
        self.itos = {int(k): v for k, v in self.word_table.items()}

        self.UNKNOWN_CHAR = self.stoi['\ue083']