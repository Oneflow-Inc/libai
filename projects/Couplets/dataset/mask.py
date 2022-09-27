import numpy as np


def make_padding_mask(q_ids, kv_ids, pad_id):
    q = (np.array(q_ids) != pad_id).reshape(-1, 1)
    kv = (np.array(kv_ids) != pad_id).reshape(1, -1)
    padding_mask = (q * kv).astype(float)
    return padding_mask


def make_sequence_mask(ids):
    seqlen = len(ids)
    sequence_mask = np.triu(np.ones((seqlen, seqlen))).transpose()
    return sequence_mask
