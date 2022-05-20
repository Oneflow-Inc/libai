import logging
import os
import pdb
import time
from enum import Enum
from typing import Optional, Union

import oneflow as flow
from filelock import FileLock
from oneflow.utils.data import Dataset

from libai.data.structures import DistTensorData, Instance

from .utils import EncodePattern
from .cner_utils import cner_convert_examples_to_features, cner_output_modes, cner_processors

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class CnerDataset(Dataset):
    def __init__(
        self,
        task_name,
        data_dir,
        tokenizer,
        max_seq_length: int = 128,
        mode: Union[str, Split] = Split.train,
        pattern: Union[str, EncodePattern] = EncodePattern.bert_pattern,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = True,
    ):
       
        self.processor = cner_processors[task_name]()
        self.output_mode = cner_output_modes[task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else data_dir,
            f"cached_{mode.value}_{tokenizer.__class__.__name__}_{max_seq_length}_{task_name}",
        )
        label_list = self.processor.get_labels()
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
         
            logger.info(f"Creating features from dataset file at {data_dir}")
            

            if mode == Split.dev:
                examples = self.processor.get_dev_examples(data_dir)
            elif mode == Split.test:
                examples = self.processor.get_test_examples(data_dir)
            else:
                examples = self.processor.get_train_examples(data_dir)

            self.features = cner_convert_examples_to_features(
                examples,
                tokenizer,
                max_length=max_seq_length,
                pattern=pattern,
                label_list=label_list,
                output_mode=self.output_mode,
            )
         
            start = time.time()
            flow.save(self.features, cached_features_file)
            logger.info(
                f"Saving features into cached file {cached_features_file} "
                f"[took {time.time() - start:.3f} s]"
            )
            

    def __len__(self):
        return len(self.features)

    # 是内置函数，可以使得外界应用DataSet类后能迭代dataset这个类，返回sample
    # 需要修改返回的格式
    def __getitem__(self, i):
        feature = self.features[i]
        return Instance(
            input_ids = DistTensorData(flow.tensor(feature.input_ids, dtype=flow.long)),
            attention_mask = DistTensorData(flow.tensor(feature.attention_mask, dtype=flow.long)),
            token_type_ids = DistTensorData(flow.tensor(feature.token_type_ids, dtype=flow.long)),
            labels = DistTensorData(flow.tensor(feature.labels, dtype=flow.long)),
        )
       

    def get_labels(self):
        return self.label_list
