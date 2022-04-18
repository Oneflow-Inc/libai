from libai.config import LazyCall

from ...modeling.vision_wrapper import VisionModel

model = LazyCall(VisionModel)(
    model_name="alexnet",
    pretrained=False,
    num_classes=1000,
)