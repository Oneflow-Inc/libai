from libai.config import LazyCall, get_config, LazyConfig

from models.model_for_training import Model
from data.ofrecord_data_utils import OFRecordDataLoader
from flowvision.transforms import RandomHorizontalFlip

model = LazyCall(Model)(
    backbone="r18",
    head="arcface",
    num_classes=94690,
    sample_rate=1.0,
    embedding_size=128,
)

dataloader = LazyCall(OFRecordDataLoader)(
    ofrecord_root="/workspace/data/insightface/ms1m-retinaface-t1/ofrecord",
    mode="train",  # "val"
    dataset_size=9469,
    batch_size=16,
    total_batch_size=16,
    data_part_num=8,
    placement=None,
    sbp=None,
    channel_last=False,
    use_gpu_decode=False,
)

train = LazyConfig.load(
    "/workspace/projects/libai/configs/common/train.py").train
optim = LazyConfig.load(
    "/workspace/projects/libai/configs/common/optim.py").optim
graph = LazyConfig.load(
    "/workspace/projects/libai/configs/common/models/graph.py").graph

# graph.enabled = False
