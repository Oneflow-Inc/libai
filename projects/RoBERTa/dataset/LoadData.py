from oneflow.utils.data import Dataset

class SST2Dataset(Dataset):
  def __init__(self, input_ids, attention_mask, labels):
    super(SST2Dataset, self).__init__()
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.labels = labels

  def __getitem__(self, key):
    return self.input_ids[key], self.attention_mask[key], self.labels[key]

  def __len__(self):
    return self.input_ids.shape[0]