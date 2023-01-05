import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from petastorm.reader import make_batch_reader
from .cached import cached_dataloader

import oneflow as flow


#@cached_dataloader(num_batches=1000)
class DLRMDataLoader(object):
    """A context manager that manages the creation and termination of a
    :class:`petastorm.Reader`.
    """

    def __init__(
        self,
        parquet_file_url_list,
        batch_size,
        num_epochs,
        shuffle_row_groups=True,
        shard_seed=1234,
        shard_count=1,
        cur_shard=0,
        num_dense_fields = 13,
        num_sparse_fields = 26,
    ):
        self.parquet_file_url_list = parquet_file_url_list
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle_row_groups = shuffle_row_groups
        self.shard_seed = shard_seed
        self.shard_count = shard_count
        self.cur_shard = cur_shard

        fields = ["label"]
        fields += [f"I{i+1}" for i in range(num_dense_fields)]
        self.I_end = len(fields)
        fields += [f"C{i+1}" for i in range(num_sparse_fields)]
        self.C_end = len(fields)
        self.fields = fields

        self._in_iter = None
        self._error = None

        self.reader = make_batch_reader(
            self.parquet_file_url_list,
            workers_count=1,
            shuffle_row_groups=self.shuffle_row_groups,
            num_epochs=self.num_epochs,
            shard_seed=self.shard_seed,
            shard_count=self.shard_count,
            cur_shard=self.cur_shard,
        )

    def __iter__(self):
        if self._error is not None:
            raise RuntimeError('Cannot start a new iteration because last time iteration failed with error {err}.'
                               .format(err=repr(self._error)))
        if self._in_iter is not None and self._in_iter == True:  # noqa: E712
            raise RuntimeError(_PARALLEL_ITER_ERROR)
        if self._in_iter is not None:
            self.reader.reset()
            print('Start a new pass of Petastorm DataLoader, reset underlying Petastorm reader to position 0.')
        self._in_iter = True

        try:
            for label, dense, sparse in self._iter_impl():
                yield {"label":label.reshape(-1, 1), "dense":dense, "sparse":sparse}
        except Exception as e:
            self._error = e
            print('Iteration on Petastorm DataLoader raise error: %s', repr(e))
            raise
        finally:
            self._in_iter = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.stop()
        self.reader.join()

    def _iter_impl(self):
        batch_size = self.batch_size
        tail = None
        for rg in self.reader:
            rgdict = rg._asdict()
            rglist = [rgdict[field] for field in self.fields]
            pos = 0
            if tail is not None:
                pos = batch_size - len(tail[0])
                tail = list(
                    [
                        np.concatenate((tail[i], rglist[i][0 : (batch_size - len(tail[i]))]))
                        for i in range(self.C_end)
                    ]
                )
                if len(tail[0]) == batch_size:
                    label = tail[0]
                    dense = tail[1 : self.I_end]
                    sparse = tail[self.I_end : self.C_end]
                    tail = None
                    yield label, np.stack(dense, axis=-1), np.stack(sparse, axis=-1)
                else:
                    pos = 0
                    continue
            while (pos + batch_size) <= len(rglist[0]):
                label = rglist[0][pos : pos + batch_size]
                dense = [rglist[j][pos : pos + batch_size] for j in range(1, self.I_end)]
                sparse = [rglist[j][pos : pos + batch_size] for j in range(self.I_end, self.C_end)]
                pos += batch_size
                yield label, np.stack(dense, axis=-1), np.stack(sparse, axis=-1)
            if pos != len(rglist[0]):
                tail = [rglist[i][pos:] for i in range(self.C_end)]


def build_criteo_dataloader(data_path, batch_size, shuffle=True):
    """Make a Criteo Parquet DataLoader.
    :return: a context manager when exit the returned context manager, the reader will be closed.
    """
    files = ["file://" + name for name in glob.glob(f"{data_path}/*.parquet")]
    files.sort()

    world_size = flow.env.get_world_size()
    batch_size_per_proc = batch_size // world_size

    return DLRMDataLoader(
        files,
        batch_size_per_proc,
        None,  # TODO: iterate over all eval dataset
        shuffle_row_groups=shuffle,
        shard_seed=1234,
        shard_count=world_size,
        cur_shard=flow.env.get_rank(),
    )

