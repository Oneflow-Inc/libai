import os
import mmap
import numpy as np
import oneflow as flow
from oneflow.utils.data import Dataset

from libai.data.structures import DistTensorData, Instance


class AlpacaDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.data = flow.load(path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return Instance(
            input_ids=DistTensorData(self.data[index]["input_ids"]),
            labels=DistTensorData(self.data[index]["labels"]),
        )


class MegatronCacheCompatibleDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        cache_path: str = "data_cache/llama2-7b_pretrain_WS8_TP2_PP1",
        split: str = "train",
        seq_length: int = 4096,
        token_dtype = None,  # default: uint16
    ):
        self.seq_length = seq_length
        self.token_size = None
        self.token_dtype = token_dtype or np.uint16  # 默认 Megatron LLaMA 使用 uint16

        for fname in os.listdir(path):
            if fname.endswith(".bin"):
                self.bin_path = os.path.join(path, fname)
                break
        else:
            raise FileNotFoundError("No .bin file found in path.")

        self._mmap_data = self._load_bin(self.bin_path)

        prefix = self._find_prefix(cache_path, split)
        self.sample_index = np.load(os.path.join(cache_path, f"{prefix}-GPTDataset-{split}-sample_index.npy"))
        self.shuffle_index = np.load(os.path.join(cache_path, f"{prefix}-GPTDataset-{split}-shuffle_index.npy"))

    def _load_bin(self, path):
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            if self.token_dtype is None:
                raise ValueError("token_dtype must be set if automatic inference not used.")
            self.token_size = np.dtype(self.token_dtype).itemsize
            return mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)

    def _find_prefix(self, cache_path, split):
        for fname in os.listdir(cache_path):
            if fname.endswith(f"GPTDataset-{split}-sample_index.npy"):
                return fname.split("-GPTDataset-")[0]
        raise FileNotFoundError(f"Cannot find index files for split={split} in {cache_path}")

    def __len__(self):
        return len(self.shuffle_index)

    def __getitem__(self, idx):
        real_idx = int(self.shuffle_index[idx])
        start_token, token_length = self.sample_index[real_idx]
        start_token = int(start_token)
        token_length = int(token_length)

        offset = start_token * self.token_size
        end = offset + self.seq_length * self.token_size

        raw = self._mmap_data[offset:end]
        tokens = np.frombuffer(raw, dtype=self.token_dtype).astype(np.int64)

        input_ids = tokens.copy()
        labels = tokens.copy()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # ignore loss

        return Instance(
            input_ids=DistTensorData(flow.tensor(input_ids, dtype=flow.long)),
            labels=DistTensorData(flow.tensor(labels, dtype=flow.long)),
        )
