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

import torch

import oneflow as flow

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.trainer import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer
# from libai.models.vit_libai import VisionTransformer

def filter_keys(key, value):
    if "norm1" in key:
        key = key.replace("norm1", "input_layernorm")
    elif "attn.qkv" in key:
        key = key.replace("attn.qkv", "self_attention.query_key_value")
        if "weight" in key:
            value = value.transpose((-1, -2))
    elif "attn.proj" in key:
        key = key.replace("attn.proj", "self_attention.dense")
        if "weight" in key:
            value = value.transpose((-1, -2))
    elif "norm2" in key:
        key = key.replace("norm2", "post_attention_layernorm")
    elif "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "mlp.dense_h_to_4h")
        if "weight" in key:
            value = value.transpose((-1, -2))
    elif "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "mlp.dense_4h_to_h")
        if "weight" in key:
            value = value.transpose((-1, -2))
    elif "head.weight" in key:
            value = value.transpose((-1, -2))
    return key, value

def load_from_torch(model, path="/home/rentianhe/code/OneFlow-Models/libai/vit_tiny_torch_weight.pth"):
    torch_dict = torch.load(path)
    parameters = torch_dict
    new_parameters = dict()
    for key, value in parameters.items():
        if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          # to global tensor
          key, val = filter_keys(key, val)
          val = flow.tensor(val).to_global(sbp=flow.sbp.broadcast, placement=flow.placement("cuda", {0: range(1)}))
          new_parameters[key] = val
    model.load_state_dict(new_parameters)
    print("successfully load pytorch weight")
    return model


class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        model = load_from_torch(model)
        return model

    def train(self):
        super().train()
        all_losses = self.storage.history("total_loss").values()
        with open("./of_loss.txt", "w") as f:
            for loss, _ in all_losses:
                f.write(str(loss) + "\n")
    
    @classmethod
    def test(cls, cfg, test_loaders, model, evaluator=None):
        return {}


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

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
