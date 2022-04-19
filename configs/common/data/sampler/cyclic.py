from libai.config import LazyCall
from libai.data.samplers import CyclicSampler

sampler = LazyCall(CyclicSampler)(
    micro_batch_size=32,
    shuffle=True
)