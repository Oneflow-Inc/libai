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


import omegaconf
from oneflow.utils.data import DataLoader
from oneflow.utils.data.dataset import ConcatDataset

from libai.utils import distributed as dist

from libai.data.data_utils.split_dataset import split_ds
from libai.data.samplers import CyclicSampler, SingleRoundSampler
from libai.data.structures import Instance


def build_megatron_gpt_train_val_test_loader(
    train_val_test_datasets,
    seed,
    train_batch_size,
    test_batch_size,
    sampler=None,
    num_workers=4,
    consumed_samples=0,
    collate_fn=None,
    **kwargs,
):
    train_dataset, val_dataset, test_dataset = train_val_test_datasets
    # train_dataset, val_dataset, test_dataset = build_megatron_datasets(
    #     data_prefix=1,
    #     data_impl=1,
    #     splits_string=1,
    #     train_valid_test_num_samples=1,
    #     seq_length=1,
    #     seed=1,
    #     skip_warmup=1,
    # )

    collate_fn = trivial_batch_collator if collate_fn is None else collate_fn

    train_loader, _, _ = build_nlp_train_loader(
        dataset=train_dataset,
        train_batch_size=train_batch_size,
        test_batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        consumed_samples=consumed_samples,
        seed=seed,
        collate_fn=collate_fn,
        shuffle=False,
    )

    valid_loader = build_nlp_test_loader(
        dataset=val_dataset,
        test_batch_size=test_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        seed=seed,
        collate_fn=collate_fn,
    )

    test_loader = build_nlp_test_loader(
        dataset=test_dataset,
        test_batch_size=test_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        seed=seed,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, test_loader


def build_nlp_train_val_test_loader(
    dataset,
    splits,
    weights,
    train_batch_size,
    test_batch_size,
    sampler=None,
    num_workers=4,
    consumed_samples=0,
    seed=0,
    collate_fn=None,
    dataset_mixer=ConcatDataset,
):
    """
    Build nlp train_val_test dataloader
    """
    # TODO: add input type, add dataset_weights sampler
    if isinstance(dataset, omegaconf.listconfig.ListConfig):
        dataset = list(dataset)
    elif not isinstance(dataset, list):
        dataset = [dataset]

    assert len(dataset) == len(splits), "datasets length must equal splits length"
    assert len(dataset) == len(weights), "datasets length must equal weights length"

    train_datasets, val_datasets, test_datasets = [], [], []
    for dst, split in zip(dataset, splits):
        train_dataset, val_dataset, test_dataset = split_ds(dst, split)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

    # [dataset, dataset] -> dataset -> dataloader
    train_dataset = dataset_mixer(train_datasets)
    val_dataset = dataset_mixer(val_datasets)
    test_dataset = dataset_mixer(test_datasets)
    # train_dataset, val_dataset, test_dataset = build_train_valid_test_datasets(
    #     data_prefix=1,
    #     data_impl=1,
    #     splits_string=1,
    #     train_valid_test_num_samples=1,
    #     seq_length=1,
    #     seed=1,
    #     skip_warmup=1,
    # )

    collate_fn = trivial_batch_collator if collate_fn is None else collate_fn

    train_loader, _, _ = build_nlp_train_loader(
        dataset=train_dataset,
        train_batch_size=train_batch_size,
        test_batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        consumed_samples=consumed_samples,
        seed=seed,
        collate_fn=collate_fn,
    )

    valid_loader = build_nlp_test_loader(
        dataset=val_dataset,
        test_batch_size=test_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        seed=seed,
        collate_fn=collate_fn,
    )

    test_loader = build_nlp_test_loader(
        dataset=test_dataset,
        test_batch_size=test_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        seed=seed,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, test_loader


def build_nlp_train_loader(
    dataset,
    train_batch_size,
    test_batch_size=None,
    sampler=None,
    num_workers=4,
    consumed_samples=0,
    seed=0,
    collate_fn=None,
    dataset_mixer=ConcatDataset,
    **kwargs,
):
    """
    Args:
        dataset: Dataset list or single dataset.
        batch_size: Batch-size for each GPU.
    """
    if isinstance(dataset, omegaconf.listconfig.ListConfig):
        dataset = list(dataset)
    elif not isinstance(dataset, list):
        dataset = [dataset]

    if len(dataset) > 1:
        dataset = dataset_mixer(dataset)
    else:
        dataset = dataset[0]

    if sampler is None:
        sampler = CyclicSampler(
            dataset=dataset,
            micro_batch_size=train_batch_size,
            shuffle=False,
            consumed_samples=consumed_samples,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
            seed=seed,
        )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        **kwargs,
    )

    return dataloader, None, None


def build_nlp_test_loader(
    dataset,
    test_batch_size,
    sampler=None,
    num_workers=4,
    seed=0,
    collate_fn=None,
):
    """
    Build nlp test dataloader
    """
    # TODO: add input type
    collate_fn = trivial_batch_collator if collate_fn is None else collate_fn
    if sampler is None:
        sampler = SingleRoundSampler(
            dataset=dataset,
            micro_batch_size=test_batch_size,
            shuffle=False,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
            seed=seed,
            drop_last=False,
        )
    test_loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate_fn
    )
    return test_loader


def build_image_train_loader(
    dataset,
    train_batch_size,
    test_batch_size=None,
    sampler=None,
    num_workers=4,
    consumed_samples=0,
    seed=0,
    collate_fn=None,
    dataset_mixer=ConcatDataset,
    mixup_func=None,
    **kwargs,
):
    """
    Args:
        dataset: Dataset list or single dataset.
        batch_size: Batch-size for each GPU.
    """
    if isinstance(dataset, omegaconf.listconfig.ListConfig):
        dataset = list(dataset)
    elif not isinstance(dataset, list):
        dataset = [dataset]

    if len(dataset) > 1:
        dataset = dataset_mixer(dataset)
    else:
        dataset = dataset[0]

    if sampler is None:
        sampler = CyclicSampler(
            dataset=dataset,
            micro_batch_size=train_batch_size,
            shuffle=True,
            consumed_samples=consumed_samples,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
            seed=seed,
        )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        **kwargs,
    )
    # Bind up mixup_func to dataloader, and this will be used in Trainer.step
    dataloader.mixup_func = mixup_func


def build_train_valid_test_data_iterators(cfg):
    """通过外部定义的build_train_valid_test_datasets_provider函数，
    1. 先计算train_val_test_num_samples；
    2. 然后传入该函数中生成数据集；
    3. 由数据集生成iterator"""

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)


def build_image_test_loader(
    dataset, test_batch_size, sampler=None, num_workers=4, seed=0, collate_fn=None, **kwargs
):
    if sampler is None:
        sampler = SingleRoundSampler(
            dataset=dataset,
            micro_batch_size=test_batch_size,
            shuffle=False,
            data_parallel_rank=dist.get_data_parallel_rank(),
            data_parallel_size=dist.get_data_parallel_size(),
            seed=seed,
            drop_last=False,
        )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        **kwargs,
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
            iter(train_dataloader) if dl_type == "single" else iter(cyclic_iter(train_dataloader))
        )
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = (
            iter(valid_dataloader) if dl_type == "single" else iter(cyclic_iter(valid_dataloader))
        )
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = (
            iter(test_dataloader) if dl_type == "single" else iter(cyclic_iter(test_dataloader))
        )
    else:
        test_data_iterator = None


def trivial_batch_collator(batch):
    assert isinstance(batch[0], Instance), "batch[0] must be `instance` for trivial batch collator"
    batch = Instance.stack(batch)
    return batch
