from libai.config import LazyCall
from projects.MOCOV3.modeling.vit_moco import VisionTransformerMoCo


model = LazyCall(VisionTransformerMoCo)()