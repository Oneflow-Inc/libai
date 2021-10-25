import oneflow as flow
from oneflow_gpt import distribute as dist


class GPTDataLoader(flow.nn.Module):
    def __init__(self, dataset, num_samples, batch_size, max_seq_length=1024, split=[1,0,0], split_index=0, num_accumulation_steps=1, seed=1234):
        super().__init__()

        batch_size = batch_size // num_accumulation_steps
        self.reader = flow.nn.GPTIndexedBinDataReader(
            data_file_prefix=dataset,
            seq_length=max_seq_length,
            num_samples=num_samples,
            batch_size=batch_size,
            dtype=flow.int64,
            shuffle=True,
            random_seed=seed,
            split_sizes=split,
            split_index=split_index,
            placement=dist.get_layer_placement(0, "cpu"),
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
        )
        self.data_decoder = DataDecoder()
        self.label_decoder = LabelDecoder()

    def forward(self):
        tokens = self.reader()
        data = self.data_decoder(tokens)
        labels = self.label_decoder(tokens)
        return data, labels


class DataDecoder(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens):
        assert tokens.ndim == 2
        return tokens.to_consistent(placement=dist.get_layer_placement(0))[:, :-1]


class LabelDecoder(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens):
        assert tokens.ndim == 2
        return tokens.to_consistent(placement=dist.get_layer_placement(-1))[:, 1:]
