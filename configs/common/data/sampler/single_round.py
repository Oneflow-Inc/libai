from libai.config import LazyCall
from libai.data.samplers import SingleRoundSampler

sampler = LazyCall(SingleRoundSampler)(
    micro_batch_size=32,
    shuffle=False
)