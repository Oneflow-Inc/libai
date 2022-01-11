
from .structures import Instance
from .samplers.distributed_sampler import TrainingSampler, InferenceSampler
import oneflow.utils.data as flowdata



def build_image_train_loader(dataset, weight, batch_size, sampler=None, num_workers=4, collate_fn=None, blendable_dataset=None):
    """ 
    dataset (list or flow.utils.data.Dataset): 一个 list of dataset 或者是单个 dataset
    augmentations: 数据增强方法，callable 
    sampler: 采样方法
    batch_size: 单卡的 batch size
    num_workers:
    collate_fn: 在 dataloader 中决定如何构建 batch 数据
    """
    if not isinstance(dataset, list):
        dataset = [dataset]

    # [dataset, dataset] -> dataset -> dataloader
    if blendable_dataset is None:
        dataset = Blendable_dataset(dataset, weight=weight) # same classes
    else:
        dataset = blendable_dataset(dataset)
    # dataset = CombineDataset(dataset) # combine classes

    if sampler is None:
        sampler = TrainingSampler()
    return flowdata.DataLoader(dataset, sampler=sampler, collate_fn=trivial_batch_collator
                               if collate_fn is None else collate_fn)


def build_image_test_loader(dataset, batch_size, sampler=None,  num_workers=4, collate_fn=None):
    """ 
    dataset: 单个 dataset
    """
    if sampler is None:
        sampler = InferenceSampler()
    return Dataloader


def build_text_train_loader(tokenizer, data_prefix=None, split=None):
    return A, B, C # train_loader, evalution_loader, test_loader


def build_text_test_loader():
    pass


def trivial_batch_collator(batch):
    assert isinstance(
        batch[0], Instance
    ), "batch[0] must be `instance` for trivial batch collator"
    batch = Instance.stack(batch)

    return batch 
