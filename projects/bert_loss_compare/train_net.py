# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

sys.path.append(".")

from data.build import build_train_valid_test_data_iterators
from tokenizer.tokenizer import setup_tokenizer
from utils.load_megatron_weight import load_megatron_bert

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import DefaultTrainer, default_setup, hooks
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer
from libai.utils.file_utils import get_data_from_cache

VOCAB_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/bert-base-chinese-vocab.txt"  # noqa
BIN_DATA_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.bin"  # noqa
IDX_DATA_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.idx"  # noqa
MODEL_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/models/megatron_bert.pt"

VOCAB_MD5 = "3b5b76c4aef48ecf8cb3abaafe960f09"
BIN_DATA_MD5 = "b842467bd5ea7e52f7a612ea6b4faecc"
IDX_DATA_MD5 = "cf5963b8543f0a7a867361eb980f0372"
MODEL_MD5 = "1ef80646d3b7a02537e85bfdcdd1eb04"


class Trainer(DefaultTrainer):

    # Remove checkpointer
    def build_hooks(self):
        ret = [
            hook
            for hook in super().build_hooks()
            if not isinstance(hook, hooks.PeriodicCheckpointer)
        ]
        return ret

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        cache_dir = os.path.join(os.getenv("LIBAI_TEST_CACHE_DIR", "./loss_align"), "models")
        if dist.is_main_process():
            # download torch weight
            get_data_from_cache(MODEL_URL, cache_dir, md5=MODEL_MD5)
        dist.synchronize()

        megatron_path = get_data_from_cache(MODEL_URL, cache_dir, md5=MODEL_MD5)
        load_megatron_bert(model, megatron_path)
        return model

    def train(self):
        super().train()
        if dist.is_main_process():
            all_losses = self.storage.history("total_loss").values()
            with open(os.path.join(self.cfg.train.output_dir, "of_loss.txt"), "w") as f:
                for loss, _ in all_losses:
                    f.write(str(loss) + "\n")

    @classmethod
    def build_train_loader(cls, cfg, tokenizer=None):
        return build_train_valid_test_data_iterators(cfg)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    cache_dir = os.path.join(os.getenv("LIBAI_TEST_CACHE_DIR", "./loss_align"), "bert_data")

    if dist.get_local_rank() == 0:
        get_data_from_cache(VOCAB_URL, cache_dir, md5=VOCAB_MD5)
        get_data_from_cache(BIN_DATA_URL, cache_dir, md5=BIN_DATA_MD5)
        get_data_from_cache(IDX_DATA_URL, cache_dir, md5=IDX_DATA_MD5)
    dist.synchronize()

    vocab_path = get_data_from_cache(VOCAB_URL, cache_dir, md5=VOCAB_MD5)
    data_prefix_path = get_data_from_cache(BIN_DATA_URL, cache_dir, md5=BIN_DATA_MD5)
    data_prefix = data_prefix_path[:-4]

    cfg.data.data_path = [data_prefix]
    cfg.data.vocab_file = vocab_path

    setup_tokenizer(cfg)

    if args.eval_only:
        tokenizer = None
        if try_get_key(cfg, "tokenization.setup", default=False):
            tokenizer = Trainer.build_tokenizer(cfg)
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        graph = Trainer.build_graph(cfg, model, is_train=False)
        test_loader = Trainer.build_test_loader(cfg, tokenizer)
        res = Trainer.test(cfg, test_loader, graph)  # noqa
        return

    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
