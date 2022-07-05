import oneflow as flow
from libai.utils import distributed as dist


class ExtendedMask(flow.nn.Module):
    def forward(self, x, input_shape=None, is_decoder=False):
        if x.dim() == 3:
            extended_mask = x.unsqueeze(1)
        elif x.dim() == 2:
            if is_decoder:
                extended_mask = create_extended_mask_for_decoder(x, input_shape)
            else:
                extended_mask = x.unsqueeze(1).unsqueeze(1)
        
        return extended_mask


def create_extended_mask_for_decoder(x, input_shape):
    batch_size, seq_len = input_shape
    seq_ids = flow.arange(seq_len, sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]), placement=x.placement)
    causal_mask = seq_ids.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1) <= seq_ids.unsqueeze(0).unsqueeze(-1)
    causal_mask = causal_mask.to(x.dtype)
    if causal_mask.shape[1] < x.shape[1]:
            prefix_seq_len = x.shape[1] - causal_mask.shape[1]
            ones = flow.ones(
                (batch_size, seq_len, prefix_seq_len),
                dtype=causal_mask.dtype, 
                sbp=causal_mask.sbp, 
                placement=causal_mask.placement
            )
            causal_mask = flow.cat(
                [
                    ones,
                    causal_mask,
                ],
                axis=-1,
            )
   
    extended_mask = causal_mask.unsqueeze(1) * x.unsqueeze(1).unsqueeze(1)
    return extended_mask


