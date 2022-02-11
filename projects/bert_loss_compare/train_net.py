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

import sys

sys.path.append(".")

from data.build import build_train_valid_test_data_iterators
from tokenizer.tokenizer import setup_tokenizer
from utils.load_megatron_weight import load_megatron_bert

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.trainer import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer


class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        load_megatron_bert(
            model,
            "/workspace/idea_model/idea_bert/megatron_model_save/bert-cn-wwm/compare_oneflow_loss_reproduce_ckpt/iter_0000100/mp_rank_00/model_optim_rng.pt",
        )
        return model

    def train(self):
        super().train()
        all_losses = self.storage.history("total_loss").values()
        with open("of_loss.txt", "w") as f:
            for loss, _ in all_losses:
                f.write(str(loss) + "\n")

    @classmethod
    def build_train_loader(cls, cfg, tokenizer=None):
        return build_train_valid_test_data_iterators(cfg)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

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
