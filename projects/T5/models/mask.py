import oneflow as flow


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
    seq_ids = flow.arange(seq_len)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_len, 1) <= seq_ids[None, :, None]
    causal_mask = causal_mask.to(x.dtype)
    causal_mask = causal_mask.to_global(sbp=x.sbp, placement=x.placement)

    if causal_mask.shape[1] < x.shape[1]:
            prefix_seq_len = x.shape[1] - causal_mask.shape[1]
            causal_mask = flow.cat(
                [
                    x.ones((batch_size, seq_len, prefix_seq_len), dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

    extended_mask = causal_mask[:, None, :, :] * x[:, None, None, :]
    return extended_mask


