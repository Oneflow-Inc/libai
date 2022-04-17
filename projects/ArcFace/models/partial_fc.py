import oneflow as flow
from libai.utils import distributed as dist


class PartialFC(flow.nn.Module):

    def __init__(
        self,
        embedding_size,
        num_classes,
        sample_rate,
    ):
        super(PartialFC, self).__init__()
        self.weight = flow.nn.Parameter(flow.empty(num_classes,
                                                   embedding_size))
        flow.nn.init.normal_(self.weight, mean=0, std=0.01)
        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_local_rank()
        self.num_local = (num_classes + self.world_size - 1) // self.world_size
        self.num_sample = int(self.num_local * sample_rate)
        self.total_num_sample = self.num_sample * self.world_size

        self.pfc = flow.nn.DistributedPariticalFCSample(self.num_sample)

    def forward(self, x, label):
        x = flow.nn.functional.normalize(x, dim=1)
        # (
        #     mapped_label,
        #     sampled_label,
        #     sampled_weight,
        # ) = self.pfc(
        #     weight=self.weight,
        #     label=label,
        #     num_sample=self.total_num_sample,
        # )
        # label = mapped_label
        # weight = sampled_weight
        weight = flow.nn.functional.normalize(self.weight, dim=1)
        x = flow.matmul(x, weight, transpose_b=True)

        return x, label