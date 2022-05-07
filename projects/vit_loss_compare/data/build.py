import omegaconf
from oneflow.utils.data import DataLoader
from oneflow.utils.data.dataset import ConcatDataset

from libai.data.build import trivial_batch_collator
from libai.utils import distributed as dist

from .compare_loss_sampler import CyclicSampler


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
    **kwargs
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

    return dataloader, None, None
