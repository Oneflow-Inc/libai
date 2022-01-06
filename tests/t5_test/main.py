from os import environ

from numpy.lib.shape_base import expand_dims
from get_megatron_t5 import get_t5_model
model = get_t5_model()

from pathlib import Path
import numpy as np
import torch

data_path = Path("/home/wang/data/t5/samples")
tokens_enc = np.load(data_path / 'tokens_enc.npy')[0: 1]
tokens_dec = np.load(data_path / 'tokens_dec.npy')[0: 1]
enc_mask = np.load(data_path / 'enc_mask.npy')[0: 1]
dec_mask = np.load(data_path / 'dec_mask.npy')[0: 1]
enc_dec_mask = np.load(data_path / 'enc_dec_mask.npy')[0: 1]

tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = map(
    torch.from_numpy, [tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask])

tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = map(
    lambda x: x.cuda(), [tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask])

model(tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask)
