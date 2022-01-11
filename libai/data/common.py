

import oneflow as flow

# cv mixture dataset

# nlp
class BlendableDataset():
    pass


# nlp (95%: train, 4%: evaluation, 1%: test)
# cv (train, test)
class SplitDataset(flow.utils.data.Dataset):
    """
    """
    def __init__(self, dataset, split_inds):
        self.split_inds = list(split_inds)
        self.wrapped_data = dataset

    def __len__(self):
        return len(self.split_inds)

    def __getitem__(self, index):
        return self.wrapped_data[self.split_inds[index]]

    @property
    def supports_prefetch(self):
        return self.wrapped_data.supports_prefetch

    def prefetch(self, indices):
        self.wrapped_data.prefetch(indices)