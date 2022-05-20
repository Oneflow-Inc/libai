import os
import sys
import unittest

import oneflow as flow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from clip.model import CLIP, ModifiedResNet, Transformer  # noqa: E402


class TestCLIP(unittest.TestCase):
    def test_modified_resnet(self):
        net = ModifiedResNet([3, 4, 6, 3], 120, 16).to_global(
            sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0])
        )

        x = flow.rand(
            32, 3, 224, 224, sbp=flow.sbp.split(0), placement=flow.placement("cuda", ranks=[0])
        )
        y = net(x)
        assert isinstance(y, flow.Tensor)

    def test_transformer(self):
        mask = flow.ones(
            12, 12, sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0])
        )
        mask = flow.tril(mask)  # zero out the lower diagonal

        # [1, 1, s, s]
        mask = mask.unsqueeze(0).unsqueeze(1).expand(16, 1, 12, 12)

        net = Transformer(128, 10, 16, mask)
        x = flow.rand(
            16, 12, 128, sbp=flow.sbp.split(0), placement=flow.placement("cuda", ranks=[0])
        )
        y = net(x)
        assert isinstance(y, flow.Tensor)

    def test_clip(self):
        # clip with resnet
        net = CLIP(
            embed_dim=10,
            # vision
            image_resolution=224,
            vision_layers=6,
            vision_width=120,
            vision_patch_size=16,
            # text
            context_length=24,
            vocab_size=3000,
            transformer_width=128,
            transformer_heads=16,
            transformer_layers=10,
        )
        img = flow.rand(
            16, 3, 224, 224, sbp=flow.sbp.split(0), placement=flow.placement("cuda", ranks=[0])
        )
        text = flow.ones(
            16,
            24,
            dtype=flow.int,
            sbp=flow.sbp.split(0),
            placement=flow.placement("cuda", ranks=[0]),
        )

        logits_img, logits_text = net(img, text)
        print(logits_img)
        print(logits_text)


if __name__ == "__main__":
    unittest.main()
