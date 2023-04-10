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

import logging
import os
import random
import sys

import numpy as np
import oneflow as flow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from onediff import OneFlowStableDiffusionPipeline  # noqa

from libai.config import LazyConfig, default_argument_parser, try_get_key  # noqa
from libai.engine import DefaultTrainer, default_setup, hooks  # noqa
from libai.engine.trainer import HookBase  # noqa
from libai.utils import distributed as dist  # noqa
from libai.utils.checkpoint import Checkpointer  # noqa

logger = logging.getLogger("libai." + __name__)


class SdCheckpointer(HookBase):
    def __init__(
        self,
        model: flow.nn.Module,
        save_path: str,
    ) -> None:
        self._model = model
        self._save_path = save_path

    def after_train(self):
        def model_to_local(model):
            model.zero_grad(set_to_none=True)
            model = model.to_global(
                sbp=flow.sbp.broadcast, placement=flow.env.all_device_placement("cpu")
            )
            return model.to_local()

        if hasattr(self._model, "lora_layers"):
            unet = model_to_local(self._model.unet)
            save_path = os.path.join(self._save_path, "model_sd_for_inference")
            logger.info(f"saving stable diffusion model to {save_path}")
            if dist.is_main_process():
                unet.save_attn_procs(save_path)
        else:
            pipeline = OneFlowStableDiffusionPipeline.from_pretrained(
                self._model.model_path,
                tokenizer=self._model.tokenizer,
                text_encoder=model_to_local(self._model.text_encoder),
                vae=model_to_local(self._model.vae),
                unet=model_to_local(self._model.unet),
            )
            save_path = os.path.join(self._save_path, "model_sd_for_inference")
            logger.info(f"saving stable diffusion model to {save_path}")
            if dist.is_main_process():
                pipeline.save_pretrained(save_path)


class Trainer(DefaultTrainer):
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),  # for beauty lr scheduler printer in `nn.Graph` mode
            SdCheckpointer(self.model, self.cfg.train.output_dir),
        ]

        if not try_get_key(self.cfg, "model.train_with_lora", default=False):
            ret.append(
                hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.train.checkpointer.period),
            )

        if self.cfg.train.evaluation.enabled:
            assert self.cfg.train.evaluation.eval_iter > 0, "run_iter must be positive number"

            def test_and_save_results():
                model = self.graph_eval if self.cfg.graph.enabled else self.model
                self._last_eval_results = self.test(self.cfg, self.test_loader, model)
                return self._last_eval_results

        if dist.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), self.cfg.train.log_period))
        return ret


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    seed_for_rank = cfg.train.seed + flow.env.get_rank()
    flow.manual_seed(seed_for_rank)
    flow.cuda.manual_seed(seed_for_rank)
    np.random.seed(seed_for_rank)
    random.seed(seed_for_rank)

    if args.fast_dev_run:
        cfg.train.train_epoch = 0
        cfg.train.train_iter = 20
        cfg.train.evaluation.eval_period = 10
        cfg.train.log_period = 1

    if args.eval_only:
        tokenizer = None
        if try_get_key(cfg, "tokenization") is not None:
            tokenizer = Trainer.build_tokenizer(cfg)
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        if try_get_key(cfg, "train.graph.enabled", default=False):
            model = Trainer.build_graph(cfg, model, is_train=False)
        test_loader = Trainer.build_test_loader(cfg, tokenizer)
        if len(test_loader) == 0:
            logger.info("No dataset in dataloader.test, please set dataset for dataloader.test")
        _ = Trainer.test(cfg, test_loader, model)
        return

    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
