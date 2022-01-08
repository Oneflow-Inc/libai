# coding=utf-8
"""
Copyright 2021 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import logging

import oneflow as flow
from libai.data.data_samplers import build_pretraining_data_loader
from libai.data.dataset_utils import train_valid_test_dataset_provider

logger = logging.getLogger(__name__)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_train_valid_test_data_iterators(cfg):
    """通过外部定义的build_train_valid_test_datasets_provider函数，
        1. 先计算train_val_test_num_samples；
        2. 然后传入该函数中生成数据集；
        3. 由数据集生成iterator"""

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    logger.info("> building train, validation, and test datasets ...")

    # Backward compatibility, assume fixed batch size.
    if cfg.train.start_iter > 0 and cfg.train.consumed_train_samples == 0:
        assert (
            cfg.train.train_samples is None
        ), "only backward compatibility support for iteration-based training"
        cfg.train.consumed_train_samples = (
            cfg.train.start_iter * cfg.train.global_batch_size
        )
    if cfg.train.start_iter > 0 and cfg.train.consumed_valid_samples == 0:
        if cfg.train.train_samples is None:
            cfg.train.consumed_valid_samples = (
                (cfg.train.start_iter // cfg.train.eval_period)
                * cfg.train.eval_iter
                * cfg.train.global_batch_size
            )

    # args中参数：
    #   global_batch_size：如果没设置，则为args.micro_batch_size * args.data_parallel_size，代表全局的 batch_size，
    #   主要作用是为了计算总的样本数或者是已经训练的样本数，每张卡上的 batch_size 是通过 micro_batch_size
    #   args.iteration是在初始化模型时setup_model_and_optimizer设置为当前已经训练的step，若为新的则为0
    #   args.train_samples为args的初始设置，如果没有则采用iter based训练，需要用args.train_iters

    # Data loader only on rank 0 of each model parallel group.
    # if mpu.get_tensor_model_parallel_rank() == 0:

    # Number of train/valid/test samples.
    if cfg.train.train_samples:
        train_samples = cfg.train.train_samples
    else:
        train_samples = cfg.train.train_iter * cfg.train.global_batch_size
    eval_iter = (
        cfg.train.train_iter // cfg.train.eval_period + 1
    ) * cfg.train.eval_iter
    test_iter = cfg.train.eval_iter
    train_val_test_num_samples = [
        train_samples,
        eval_iter * cfg.train.global_batch_size,
        test_iter * cfg.train.global_batch_size,
    ]
    logger.info(" > datasets target sizes (minimum size):")
    logger.info("    train:      {}".format(train_val_test_num_samples[0]))
    logger.info("    validation: {}".format(train_val_test_num_samples[1]))
    logger.info("    test:       {}".format(train_val_test_num_samples[2]))

    # Build the datasets.
    train_ds, valid_ds, test_ds = train_valid_test_dataset_provider(
        cfg, train_val_test_num_samples
    )

    # Build dataloders.
    train_dataloader = build_pretraining_data_loader(
        cfg, train_ds, cfg.train.consumed_train_samples
    )
    valid_dataloader = build_pretraining_data_loader(
        cfg, valid_ds, cfg.train.consumed_valid_samples
    )
    test_dataloader = build_pretraining_data_loader(cfg, test_ds, 0)

    # Flags to know if we need to do training/validation/testing.
    do_train = train_dataloader is not None and cfg.train.train_iter > 0
    do_valid = valid_dataloader is not None and cfg.train.eval_iter > 0
    do_test = test_dataloader is not None and cfg.train.eval_iter > 0
    # Need to broadcast num_tokens and num_type_tokens.
    flags = flow.tensor(
        [int(do_train), int(do_valid), int(do_test)], dtype=flow.long, device="cuda"
    )
    # flags = torch.cuda.LongTensor(
    #     [int(do_train), int(do_valid), int(do_test)])
    # else:
    #     flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    # torch.distributed.broadcast(flags,
    #                             mpu.get_tensor_model_parallel_src_rank(),  # 获取当前全局rank对应的tp组的第一个local rank
    #                             group=mpu.get_tensor_model_parallel_group())
    cfg.train.do_train = flags[0].item()
    cfg.train.do_valid = flags[1].item()
    cfg.train.do_test = flags[2].item()

    # Build iterators.
    dl_type = cfg.data.dataloader_type
    assert dl_type in ["single", "cyclic"]

    if train_dataloader is not None:
        train_data_iterator = (
            iter(train_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(train_dataloader))
        )
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = (
            iter(valid_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(valid_dataloader))
        )
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = (
            iter(test_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(test_dataloader))
        )
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
