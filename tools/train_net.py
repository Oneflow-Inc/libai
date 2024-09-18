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
import random
import importlib

import numpy as np
import oneflow as flow
import oneflow_xpu

import libai.utils.distributed as dist
from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer
from configs.loader_mapping import loader_mapping_models as mapping


def nan_tensors(tensors):
    for tensor in tensors:
        if not isinstance(tensor, flow.Tensor):
            continue
        array = tensor.numpy()
        if np.any(np.isinf(array)) or np.any(np.isnan(array)):
            return True
    return False


def create_forward_hook(module_name):
    def save_output(module, input, output):
        print(f"forward {module_name=} input_nan={nan_tensors(input)} output_nan={nan_tensors(output)}")
    return save_output


def create_backward_hook(module_name):
    def save_output(module, input, output):
        print(f"backward {module_name=} input_nan={nan_tensors(input)} output_nan={nan_tensors(output)}")
    return save_output


def build_model(cfg):
    model_arguments=mapping[cfg.cfg.model_type]
    Loader = getattr(
        importlib.import_module(model_arguments['loader_prefix']),
        model_arguments['huggingface_loader'],
    )
    model_loader = Loader(
        cfg,
        cfg.cfg,
        cfg.cfg.pretrained_model_path,
    )
    model = model_loader.load()
    return model

class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        assert try_get_key(cfg, "model") is not None, "cfg must contain `model` namespace"
        # Set model fp16 option because of embedding layer `white_identity` manual
        # insert for amp training if provided.
        if try_get_key(cfg.model, "cfg.amp_enabled") is not None:
            cfg.model.cfg.amp_enabled = cfg.train.amp.enabled and cfg.graph.enabled
        # In case some model define without cfg keyword.
        elif try_get_key(cfg.model, "amp_enabled") is not None:
            cfg.model.amp_enabled = cfg.train.amp.enabled and cfg.graph.enabled
        model = build_model(cfg.model)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        model._apply(dist.convert_to_distributed_default_setting)

        if cfg.train.train_with_fp16:
            model = model.to(flow.float16)
            flow.cuda.empty_cache()
        '''for param in model.named_parameters():
            print(param[1].dtype)'''
        
        for module_name, module in model.named_modules():
            if module_name:
                module.register_forward_hook(create_forward_hook(module_name))
                module.register_full_backward_hook(create_backward_hook(module_name))
        return model

def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.fast_dev_run:
        cfg.train.train_epoch = 0
        cfg.train.train_iter = 20
        cfg.train.evaluation.eval_period = 10
        cfg.train.log_period = 1

    if args.eval_only:
        tokenizer = None
        if try_get_key(cfg, "tokenization") is not None:
            tokenizer = DefaultTrainer.build_tokenizer(cfg)
        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        if try_get_key(cfg, "graph.enabled", default=False):
            model = DefaultTrainer.build_graph(cfg, model, is_train=False)
        test_loader = DefaultTrainer.build_test_loader(cfg, tokenizer)
        if len(test_loader) == 0:
            logger = logging.getLogger(__name__)
            logger.info("No dataset in dataloader.test, please set dataset for dataloader.test")
        _ = DefaultTrainer.test(cfg, test_loader, model)
        return

    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
