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

import oneflow as flow

sys.path.append(".")
from libai.config import LazyConfig, default_argument_parser
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer

from libai.trainer import DefaultTrainer, default_setup


def get_batch(data_iterator):
    """Build the batch for Bert model."""

    assert data_iterator is not None, "data iterator is None!"
    data = next(data_iterator)

    input_placement = dist.get_layer_placement(0)
    label_placement = dist.get_layer_placement(-1)
    sbp = dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])

    def to_consistent(tensor, placement):
        tensor = tensor.to_consistent(placement, sbp)
        return tensor

    # Unpack.
    tokens = to_consistent(data["text"].long(), input_placement)
    types = to_consistent(data["types"].long(), input_placement)
    padding_mask = to_consistent(data["padding_mask"].long(), input_placement)
    sentence_order = to_consistent(data["is_random"].long(), label_placement)
    loss_mask = to_consistent(data["loss_mask"].float(), label_placement)
    lm_labels = to_consistent(data["labels"].long(), label_placement)

    return tokens, padding_mask, types, sentence_order, lm_labels, loss_mask


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # load megatron weight
        from libai.utils.load_megatron_weight import load_megatron_bert

        load_megatron_bert(
            self.model,
            "/workspace/idea_model/idea_bert/megatron_model_save/bert-cn-wwm/compare_oneflow_loss_reproduce_ckpt/iter_0000100/mp_rank_00/model_optim_rng.pt",
        )

    def run_step(self):
        return super().run_step(get_batch)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        graph = Trainer.build_graph(cfg, model, is_train=False)
        res = Trainer.test(cfg, graph)

    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
