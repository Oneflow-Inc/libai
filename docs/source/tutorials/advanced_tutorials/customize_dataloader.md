# How to Customize Dataloader

Dataloader is the component that provides data to models. Dataloader usually (but not necessarily) takes raw information from [write dataloaders](https://libai.readthedocs.io/en/latest/tutorials/basics/Write_Dataloaders.html), and processes them into the format needed by the model.

## How the Existing Dataloader Works 

LiBai contains a built-in data loading pipeline. It's beneficial to understand how it works, in case you need to write a custom one.

LiBai provides some functions [build_{image,nlp}_{train,test}_loader](https://libai.readthedocs.io/en/latest/modules/libai.data.html#libai.data.build.build_nlp_train_loader) that create a default dataloader from a given config. Here is how `build_{image,nlp}_{train,test}_loader` work:

1. It instantiates the `list[flow.utils.Dataset]` (e.g., `BertDataset`) by loading some dataset items with lightweight format. These dataset items are not yet ready to be used by the model (e.g., images are not loaded into memory, random augmentation have not been applied, etc.). 

2. The output format of dataset (`__getitem__(...)`) must be a dict whose keys must be consistent with argument names of the dataloader's consumer (usually the `model.forward(...)`). The role of the process is to transform the lightweight representation of a dataset item into a format that is ready for the model to consume (including, e.g., read images, perform random data augmentation and convert to oneflow Tensors). If you would like to perform custom transformations to data, you often want to rewrite it. Details about the dataset format can be found in [write dataloaders](https://libai.readthedocs.io/en/latest/tutorials/basics/Write_Dataloaders.html).

3. The outputs of the dataset are simply batched with the following function.

```python
def trivial_batch_collator(batch):
    assert isinstance(batch[0], Instance), "batch[0] must be `instance` for trivial batch collator"
    batch = Instance.stack(batch)
    return batch
```

4. This batched data is the output of the dataloader. Typically, it's also the input of `get_batch`. After `get_batch(...)`, it becomes the input of `model.forward()`. `get_batch` simply changes the local tensors to global tensors with the given `sbp` and `placement` meta information.


```python
@classmethod
def get_batch(cls, data, mixup_func = None):
    ...
    ret_dict = {}
    for key, value in data.get_fields().items():
        value.to_global()
        ret_dict[key] = value.tensor
    return ret_dict
```


## Use Custom Dataloader

If you use `DefaultTrainer`, you can overwrite its `build_train_loader` method to use your own dataloader which can be implemented with any tools you like. But you need to make sure that each rank is reading the data correctly under different parallelism circumstances.

Then you need to overwrite `get_batch` method. `data` argument in `get_batch` is the output of your dataloader. You need to change the local tensors to global tensors manually, which means you should set the `sbp` and `placement` correctly.

Here is an example. Process of rank0 gets all data and redistributes them into the other ranks.

```python
@classmethod
def get_batch(cls, data, mixup_func=None):
    if data is None: 
        # not rank0, set placeholders for data
        # Note: make sure imgs and labels have the same shape and dtype on all ranks
        imgs = flow.empty(16, 3, 224, 224, dtype=flow.float32)
        labels = flow.empty(16, dtype=flow.int64)
    else: 
        # rank0
        imgs, labels = data
    dist.synchronize()

    imgs = imgs.to_global(spb=flow.sbp.broadcast, placement=flow.env.all_device_placement("cuda"))
    imgs = imgs.to_global(
        spb=dist.get_nd_sbp([flow.sbp.split(0),
                             flow.sbp.broadcast]),
        placement=dist.get_layer_placement(0))

    labels = labels.to_global(spb=flow.sbp.broadcast, placement=flow.env.all_device_placement("cuda"))
    labels = labels.to_global(
        spb=dist.get_nd_sbp([flow.sbp.split(0),
                             flow.sbp.broadcast]),
        placement=dist.get_layer_placement(-1))
    return {
        "images": imgs,
        "labels": labels
    }
```
