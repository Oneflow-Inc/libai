import os
import sys
import unittest

import numpy as np
import oneflow as flow
import torch
from torch.nn.functional import multi_head_attention_forward as multi_head_attention_forward_torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from clip.ops import multi_head_attention_forward as multi_head_attention_forward_flow  # noqa: E402


class TestMultiHeadAttention(unittest.TestCase):
    def test_with_torch(self):
        k_proj_weight = np.random.normal(size=(32, 32))
        k_proj_bias = np.random.normal(size=(32))

        q_proj_weight = np.random.normal(size=(32, 32))
        q_proj_bias = np.random.normal(size=(32))

        v_proj_weight = np.random.normal(size=(32, 32))
        v_proj_bias = np.random.normal(size=(32))

        c_proj_weight = np.random.normal(size=(64, 32))
        c_proj_bias = np.random.normal(size=(64))

        x = np.random.normal(size=(65, 16, 32))

        x_torch = torch.from_numpy(x)
        torch_out, _ = multi_head_attention_forward_torch(
            query=x_torch,
            key=x_torch,
            value=x_torch,
            embed_dim_to_check=x_torch.shape[-1],
            num_heads=8,
            q_proj_weight=torch.from_numpy(q_proj_weight),
            k_proj_weight=torch.from_numpy(k_proj_weight),
            v_proj_weight=torch.from_numpy(v_proj_weight),
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [
                    torch.from_numpy(q_proj_bias),
                    torch.from_numpy(k_proj_bias),
                    torch.from_numpy(v_proj_bias),
                ]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=torch.from_numpy(c_proj_weight),
            out_proj_bias=torch.from_numpy(c_proj_bias),
            use_separate_proj_weight=True,
            training=True,
            need_weights=False,
        )

        x_flow = flow.from_numpy(x).cuda()
        flow_out, _ = multi_head_attention_forward_flow(
            query=x_flow,
            key=x_flow,
            value=x_flow,
            embed_dim_to_check=x_flow.shape[-1],
            num_heads=8,
            q_proj_weight=flow.from_numpy(q_proj_weight).cuda(),
            k_proj_weight=flow.from_numpy(k_proj_weight).cuda(),
            v_proj_weight=flow.from_numpy(v_proj_weight).cuda(),
            in_proj_weight=None,
            in_proj_bias=flow.cat(
                [
                    flow.from_numpy(q_proj_bias).cuda(),
                    flow.from_numpy(k_proj_bias).cuda(),
                    flow.from_numpy(v_proj_bias).cuda(),
                ]
            ),
            bias_k=None,
            bias_v=None,
            dropout_p=0,
            out_proj_weight=flow.from_numpy(c_proj_weight).cuda(),
            out_proj_bias=flow.from_numpy(c_proj_bias).cuda(),
            use_separate_proj_weight=True,
            training=True,
            need_weights=False,
        )

        assert np.allclose(torch_out.numpy(), flow_out.numpy())


if __name__ == "__main__":
    unittest.main()
