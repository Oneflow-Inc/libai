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

import oneflow as flow
import sys
sys.path.append(".")
from libai.trainer import DefaultTrainer, default_setup
from libai.trainer.trainer import HookBase

# NOTE: Temporarily use yacs as config 
from yacs.config import CfgNode as CN
from tests.layers.test_trainer_model import build_model, build_graph


def setup():
    """
    Create configs and perform basic setups.
    """
    
    cfg = CN()
    cfg.output_dir = "./demo_output"
    cfg.load = None # "./demo_output2/model_0000999"
    cfg.start_iter = 0
    cfg.train_iters = 6000
    cfg.global_batch_size = 64
    cfg.save_interval = 1000
    cfg.log_interval = 20
    cfg.nccl_fusion_threshold_mb = 16
    cfg.nccl_fusion_max_ops = 24
    cfg.mode = "graph"
    cfg.data_parallel_size = 1
    cfg.micro_batch_size = 32

    default_setup(cfg)
    return cfg

class DemoTrianer(DefaultTrainer):
    @staticmethod
    def get_batch(data_interator, mode):
        assert mode in ["eager", "graph"]
        # data = next(data_interator)
        data = flow.randn(32, 512).to("cuda")
        if mode == "graph":
            data = data.to_consistent(sbp=flow.sbp.split(0), placement = flow.env.all_device_placement("cuda"))
        return (data, )
    
    def run_step(self):
        return super().run_step(self.get_batch)
    
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            flow.nn.Module:
        It now calls :func:`libai.layers.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        return model

    @classmethod
    def build_graph(cls, cfg, model, optimizer, lr_scheduler):
        return build_graph(cfg, model, optimizer, lr_scheduler)


def main():
    cfg = setup()

    trainer = DemoTrianer(cfg)

    # trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    main()