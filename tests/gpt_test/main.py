from pathlib import Path
from libai.models import GPTModel
from libai.utils.checkpoint import Checkpointer
import oneflow as flow
import torch

if __name__ == '__main__':
    HIDDEN_SIZE=384
    VOCAB_SIZE=50304
    NUM_ATTENTION_HEADS=16
    FFN_HIDDEN_SIZE=1536
    ENCODER_SEQ_LENGTH=512
    DECODER_SEQ_LENGTH=256

    flow_gpt = GPTModel(
        num_layers=6,
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        max_seq_length=DECODER_SEQ_LENGTH,
        embedding_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        apply_query_key_layer_scaling=False,
    )
    flow_gpt.eval()

    from get_megatron_gpt import get_gpt_model
    from utils import load_megatron_weight, get_sample, get_random_sample
    
    megatron_gpt = get_gpt_model()

    if Path('megatron_gpt.pth').exists():
        for k, v in torch.load('megatron_gpt.pth').items():
            megatron_gpt.state_dict()[k].copy_(v)

    if not Path('flow_gpt.f/').exists():
        load_megatron_weight(flow_gpt, megatron_gpt)
    else:
        flow_gpt.load_state_dict(flow.load('flow_gpt.f', 0))

    flow.save(flow_gpt.state_dict(), "flow_gpt.f", global_dst_rank=0)
    torch.save(megatron_gpt.state_dict(), "megatron_gpt.pth")

    
    FLOW_FM = None
    def flow_hook(module, inp, out):
        global FLOW_FM
        if FLOW_FM is None:
            FLOW_FM = out
    

    MEGA_FM = None
    def mega_hook(module, inp, out):
        global MEGA_FM
        if MEGA_FM is None:
            MEGA_FM = out
    

    class Graph(flow.nn.Graph):
        def __init__(self) -> None:
            super().__init__()
            self.model = flow_gpt
        
        def build(self, tokens_enc):
            return self.model(tokens_enc)


    flow_gpt = Graph()
    tokens_enc, position_ids, enc_mask = get_sample('flow')
    with flow.no_grad():
        flow_output = flow_gpt(
            tokens_enc[:, :DECODER_SEQ_LENGTH],
        )
        
    
    tokens_enc, position_ids, enc_mask = get_sample('torch')
    import torch
    with torch.no_grad():
        megatron_output = megatron_gpt(
            tokens_enc[:, :DECODER_SEQ_LENGTH],
            position_ids[:, :DECODER_SEQ_LENGTH],
            enc_mask[..., :DECODER_SEQ_LENGTH, :DECODER_SEQ_LENGTH],
        )
    import numpy as np
    print('max diff: ', np.max(np.max(flow_output.numpy() - megatron_output.detach().cpu().numpy())))


    # diff_results = []
    # for i in range(100):
    #     flow_tensor, torch_tensor = get_random_sample(VOCAB_SIZE)
    #     tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = flow_tensor
    #     with flow.no_grad():
    #         flow_output = flow_t5(
    #             tokens_enc,
    #             tokens_dec,
    #             enc_mask,
    #             dec_mask,
    #             enc_dec_mask,
    #         )

    #     tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask = torch_tensor
    #     with torch.no_grad():
    #         megatron_output = megatron_t5(
    #             tokens_enc,
    #             tokens_dec,
    #             enc_mask,
    #             dec_mask,
    #             enc_dec_mask,
    #         )
    #     print(f'{i}th max diff: ', np.max(np.max(flow_output.detach().numpy() - megatron_output.detach().cpu().numpy())))
    #     diff_results.append(np.max(np.max(flow_output.numpy() - megatron_output.detach().cpu().numpy()))) 
    # print('avg max diff: ', np.average(diff_results))
