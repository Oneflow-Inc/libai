import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import oneflow as flow
from .petastorm_dataloader import DataLoader
from petastorm.reader import make_batch_reader


def collate_fn(data):
    df = pd.DataFrame(data)
    label = df['label'].to_numpy().reshape(-1, 1)
    dense = df[df.columns[1:14]].to_numpy()
    sparse = df[df.columns[14:]].to_numpy()
    return {"label":label, "dense":dense, "sparse":sparse}


def build_criteo_dataloader(data_path, batch_size, shuffle=True):
    """Make a Criteo Parquet DataLoader.
    :return: a context manager when exit the returned context manager, the reader will be closed.
    """
    files = ["file://" + name for name in glob.glob(f"{data_path}/*.parquet")]
    files.sort()

    world_size = flow.env.get_world_size()
    batch_size_per_proc = batch_size // world_size

    reader = make_batch_reader(
        files,
        workers_count=1,
        shuffle_row_groups=shuffle,
        num_epochs=1,
        shard_seed=1234,
        shard_count=world_size,
        cur_shard=flow.env.get_rank(),
    )
    return DataLoader(reader, collate_fn=collate_fn, batch_size=batch_size_per_proc) 

