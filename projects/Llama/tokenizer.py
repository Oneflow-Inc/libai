import oneflow as flow
import sentencepiece as spm

import libai.utils.distributed as dist


class LlamaTokenizer:
    def __init__(
        self,
        pretrained_model_path,
        bos_token_id="<s>",
        eos_token_id="</s>",
        pad_token_id="<unk>",
    ):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(pretrained_model_path)

        self.bos_token_id = self.sp_model.bos_id() if self.sp_model.bos_id() else bos_token_id
        self.eos_token_id = self.sp_model.eos_id() if self.sp_model.eos_id() else eos_token_id
        self.pad_token_id = self.sp_model.pad_id() if self.sp_model.pad_id() else pad_token_id

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    def encode(self, text):
        tokens = self.sp_model.encode(text)
        return tokens

    def tokenize(self, text, add_bos=False, add_eos=False, **kwargs):
        if isinstance(text, str):
            tokens = [self.sp_model.encode(text)]

        if isinstance(text, list):
            tokens = [self.sp_model.encode(s) for s in text]

        if add_bos:
            tokens = [[self.bos_token_id] + token for token in tokens]
        if add_eos:
            tokens = [[self.eos_token_id] + token for token in tokens]

        sbp = kwargs.get("sbp", dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]))
        placement = kwargs.get("placement", flow.placement("cuda", [0]))
        return_token_ids = flow.tensor(tokens, sbp=sbp, placement=placement, dtype=flow.long)
        return return_token_ids

    def decode(self, tokens):
        if isinstance(tokens, flow.Tensor):
            tokens = tokens.tolist()
        return self.sp_model.decode(tokens)

    def convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index)
