import os

import oneflow as flow
from dataset.mask import make_padding_mask, make_sequence_mask
from oneflow.utils.data import Dataset
from tokenizer.tokenizer import CoupletsTokenizer

from libai.data.structures import DistTensorData, Instance


class CoupletsDataset(Dataset):
    def __init__(self, path, is_train=True, maxlen=64):
        if is_train:
            datapath = os.path.join(path, "train")
        else:
            datapath = os.path.join(path, "test")

        src = []
        with open(f"{datapath}/in.txt", "r") as f_src:
            for line in f_src.readlines():
                src.append(line.strip("\n"))
        tgt = []
        with open(f"{datapath}/out.txt", "r") as f_tgt:
            for line in f_tgt.readlines():
                tgt.append(line.strip("\n"))
        self.data = list(zip(src, tgt))
        self.tokenizer = CoupletsTokenizer(f"{path}/vocab.txt")
        self.maxlen = maxlen
        self.unk_id = self.tokenizer.unk_id
        self.pad_id = self.tokenizer.pad_id
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id

    def __len__(self):
        return len(self.data)

    def text2ids(self, text):
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[: self.maxlen - 2]
        ids = [self.bos_id] + ids + [self.eos_id]
        ids = ids + [self.pad_id] * (self.maxlen - len(ids))
        return ids

    def __getitem__(self, index):
        sample = self.data[index]
        src_ids = self.text2ids(sample[0])
        tgt_ids = self.text2ids(sample[1])
        encoder_self_attn_mask = make_padding_mask(src_ids, src_ids, self.pad_id)
        decoder_self_attn_mask = make_padding_mask(
            tgt_ids, tgt_ids, self.pad_id
        ) * make_sequence_mask(tgt_ids)
        cross_attn_mask = make_padding_mask(tgt_ids, src_ids, self.pad_id)

        return Instance(
            encoder_input_ids=DistTensorData(flow.tensor(src_ids, dtype=flow.long)),
            decoder_input_ids=DistTensorData(flow.tensor(tgt_ids, dtype=flow.long)),
            encoder_attn_mask=DistTensorData(flow.tensor(encoder_self_attn_mask, dtype=flow.long)),
            decoder_attn_mask=DistTensorData(flow.tensor(decoder_self_attn_mask, dtype=flow.long)),
            encoder_decoder_attn_mask=DistTensorData(flow.tensor(cross_attn_mask, dtype=flow.long)),
        )
