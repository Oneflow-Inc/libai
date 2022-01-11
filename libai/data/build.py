
from .structures import Instance
from .samplers.distributed_sampler import TrainingSampler, InferenceSampler
import oneflow.utils.data as flowdata
from libai.config import instantiate


def build_nlp_dataset(dataset, split):
    train_dataset, valid_dataset, test_dataset = split_ds(total_dataset, split)
    return train_dataset, valid_dataset, test_dataset


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


def build_nlp_train_val_test_loader(cfg.dataloader.train.datasets, weight, batch_size, sampler=None, num_workers=4, collate_fn=None, blendable_dataset=Blendable_dataset):
    if not isinstance(cfg.dataloader.train.dataset, list):
        dataset = [dataset]
        
    train_datasets, val_datasets, test_datasets = [], [], []
    for dst in cfg.dataloader.train.datasets:
        train_dataset, val_dataset, test_dataset =  instantiate(dst)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

    # [dataset, dataset] -> dataset -> dataloader
    train_dataset = Blendable_dataset(train_datasets, weight=weight) # same classes
    val_dataset = Blendable_dataset(val_datasets, weight=weight) 
    test_dataset = Blendable_dataset(test_datasets, weight=weight) 
    
    train_loader = flowdata.DataLoader(train_dataset, sampler=sampler, collate_fn=trivial_batch_collator
                    if collate_fn is None else collate_fn)

    evalution_loader =  flowdata.DataLoader(val_dataset, sampler=sampler, collate_fn=trivial_batch_collator
                    if collate_fn is None else collate_fn)

    test_loader = flowdata.DataLoader(test_dataset, sampler=sampler, collate_fn=trivial_batch_collator
                    if collate_fn is None else collate_fn)

    # 最后在trainer里面 把test_loader和evaluation_loader加入到 trainer中的self.test_loaders里面
    return train_loader, evalution_loader, test_loader


def build_nlp_test_loader(dataset, batch_size, sampler=None, num_workers=4, collate_fn=None,):
    dataset = instantiate(dataset)

    if sampler is None:
        sampler = InferenceSampler()
    return Dataloader


def trivial_batch_collator(batch):
    assert isinstance(
        batch[0], Instance
    ), "batch[0] must be `instance` for trivial batch collator"
    batch = Instance.stack(batch)
    return batch 
