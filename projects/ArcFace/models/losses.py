import oneflow as flow
from oneflow import nn


class CrossEntropyLoss_sbp(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss_sbp, self).__init__()

    def forward(self, logits, label):
        loss = flow._C.sparse_softmax_cross_entropy(logits, label)
        loss = flow.mean(loss)
        return loss


def get_loss(name):
    # if name == "cosface":
    #     return CosFace()
    # elif name == "arcface":
    #     return ArcFace()
    # else:
    #     raise ValueError()
    if name == "cosface":
        margin_softmax = flow.nn.CombinedMarginLoss(1, 0.0, 0.4).to("cuda")
    else:
        margin_softmax = flow.nn.CombinedMarginLoss(1, 0.5, 0.0).to("cuda")
    return margin_softmax


class CosFace(nn.Module):

    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = flow.where(label != -1)[0]
        m_hot = flow.zeros(index.size()[0],
                           cosine.size()[1],
                           device=cosine.device)

        m_hot = flow.scatter(m_hot, 1, label[index, None], self.m)
        cosine = cosine[index] - m_hot

        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):

    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: flow.Tensor, labels):
        import ipdb
        ipdb.set_trace()
        index = flow.where(labels != -1)[0]
        m_hot = flow.zeros(
            index.size()[0],
            cosine.size()[1],
            #    device=cosine.device,
            placement=cosine.placement,
            sbp=cosine.sbp)
        m_hot = flow.scatter(m_hot, 1, labels[index, None], self.m)
        # m_hot.scatter_(1, labels[index, None], self.m)
        cosine = cosine.acos()
        # cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine
