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

from utils.load_torch_weight import load_from_torch

import libai.utils.distributed as dist
from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.trainer import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer
from libai.utils.file_utils import get_data_from_cache

DATA_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/cifar10/cifar-10-python.tar.gz"
MODEL_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/models/torch_vit_tiny_weight_cifar10.pth"

DATA_MD5 = "c58f30108f718f92721af3b95e74349a"
MODEL_MD5 = "9e9edd3782d0c9dcb75d86f53800e3f9"


class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        cache_dir = os.path.join(os.getenv("LIBAI_TEST_CACHE_DIR", "./loss_align"), "models")
        torch_weight_path = get_data_from_cache(MODEL_URL, cache_dir, md5=MODEL_MD5)
        model = load_from_torch(model, path=torch_weight_path)
        print("Successfully load weight")
        return model

    def train(self):
        super().train()
        all_losses = self.storage.history("total_loss").values()
        with open(os.path.join(self.cfg.train.output_dir, "./of_loss.txt"), "w") as f:
            print("write loss")
            for loss, _ in all_losses:
                f.write(str(loss) + "\n")

    @classmethod
    def test(cls, cfg, test_loaders, model, evaluator=None):
        return {}


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    cache_dir = os.path.join(os.getenv("LIBAI_TEST_CACHE_DIR", "./loss_align"), "vit_data")

    if dist.get_local_rank() == 0:
        get_data_from_cache(DATA_URL, cache_dir, DATA_MD5)
    dist.synchronize()
    data_path = get_data_from_cache(DATA_URL, cache_dir, md5=DATA_MD5)
    cfg.dataloader.train.dataset[0].root = "/".join(data_path.split("/")[:3])

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
