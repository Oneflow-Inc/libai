from libai.config import LazyCall
from libai.models import AlexNet

model = LazyCall(AlexNet)(
    num_classes=1000,
)
