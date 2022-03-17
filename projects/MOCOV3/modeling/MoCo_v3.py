import math

import oneflow as flow
import oneflow.nn as nn

from libai.utils.distributed import get_world_size
from libai.layers import Linear


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, momentum_encoder, dim=256, mlp_dim=4096, T=1.0, m=0.99, max_iter=300):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """ 
        super(MoCo, self).__init__()

        self.T = T
        self.m = m
        # build encoders
        self.base_encoder = base_encoder
        self.momentum_encoder = momentum_encoder
        self.base_encoder.num_classes = dim
        self.momentum_encoder.num_classes = dim
        self.max_iter = max_iter

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient


    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(Linear(dim1, dim2,  bias=False)) # libai
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN

                # mlp.append(nn.BatchNorm1d(dim2, affine=False)) # for oneflow, it returns bug: gamma need to be tensor, not NoneType
                mlp.append(nn.BatchNorm1d(dim2, affine=True))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @flow.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)  # for graph mode ->bug: AttributeError: 'TensorBlock' object has no attribute 'data'. It works well on eager mode

    def contrastive_loss(self, q, k):

        # normalize
        q = nn.functional.normalize(q, dim=1) 
        k = nn.functional.normalize(k, dim=1)
        
        # gather all targets
        k = concat_all_gather(k).to_global(sbp=q.sbp, placement=q.placement)

        # Einstein sum is more intuitive
        logits = flow.matmul(q, k.T) / self.T # Equals to flow.einsum('nc,mc->nm', [q, k]) / self.T
        # logits = flow.einsum('nc,mc->nm', [q, k]) / self.T  # Oneflow does not support Einstein sum currently
        N = logits.shape[0] // get_world_size() 
        labels = (flow.arange(N, dtype=flow.long) + N * flow.env.get_rank()).cuda().to_global(sbp=flow.sbp.split(0), placement=logits.placement)

        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def adjust_moco_momentum(self, cu_iter, m):
        """Adjust moco momentum based on current epoch"""
        m = 1. - 0.5 * (1. + math.cos(math.pi * cu_iter / self.max_iter)) * (1. - m)
        return m

    def forward(self, images, labels=None, cu_iter=0, m=0.99):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        if self.training:
            [x1, x2] = flow.chunk(images, 2, dim=1)
            # compute features
            q1 = self.predictor(self.base_encoder(x1)['prediction_scores'])
            q2 = self.predictor(self.base_encoder(x2)['prediction_scores'])

            m = self.adjust_moco_momentum(cu_iter, m) # update the moco_momentum

            with flow.no_grad():  # no gradient
                self._update_momentum_encoder(m)  # update the momentum encoder

                # compute momentum features as targets
                k1 = self.momentum_encoder(x1)['prediction_scores']
                k2 = self.momentum_encoder(x2)['prediction_scores']

            return {"losses": self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)}, {"m":m}
        else:
            return self.base_encoder(images)



class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[0] # linear.weight.T
        
        # del -> bug: AttributeError: head
        # del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer
        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@flow.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: flow.distributed.all_gather has no gradient.
    """

    tensor = tensor.to_local()

    tensors_gather = [flow.ones_like(tensor)
        for _ in range(get_world_size())]

    # flow.distributed.all_gather(tensors_gather, tensor, async_op=False)
    flow.comm.all_gather(tensors_gather, tensor) # all tensors are consistent here, thus .device is unavailable

    output = flow.cat(tensors_gather, dim=0)
    return output