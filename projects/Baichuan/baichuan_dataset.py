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
