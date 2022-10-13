import oneflow as flow
from oneflow import nn
from libai.config import LazyConfig, LazyCall
from omegaconf import OmegaConf
from libai.engine import DefaultTrainer
from libai.tokenizer import T5Tokenizer
from oneflow_onnx.oneflow2onnx.util import export_onnx_model, convert_to_onnx_and_check

def convert_to_local_model(t):
    if t.is_global:
        return t.to_local()
    else:
        return t

def get_model(config_file):
    cfg = LazyConfig.load(config_file)

    cfg.model.cfg.mlp_type = "t5"
    cfg.model.cfg.pretrained_model_path = None
    cfg.dataloader = None
    cfg.tokenization = None

    model = DefaultTrainer.build_model(cfg)
    # model._apply(convert_to_local_model)

    return model

class t5Graph(nn.Graph):
    def __init__(self, eager_model):
        super().__init__()
        self.model = eager_model

    def build(
            self, 
            encoder_input_ids, 
            encoder_attn_mask, 
            decoder_input_ids, 
            decoder_attn_mask, 
            encoder_decoder_attn_mask
        ):
        out = self.model(
            encoder_input_ids, 
            encoder_attn_mask, 
            decoder_input_ids, 
            decoder_attn_mask, 
            encoder_decoder_attn_mask
        )
        return out["prediction_scores"]

model = get_model("projects/MT5/configs/mt5_pretrain.py")
model.eval()

t5_graph = t5Graph(model)
# Build the static graph model
encoder_input_ids = flow.ones(1, 5, dtype=flow.int64, sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0]))
encoder_attn_mask = flow.ones(1, 3, dtype=flow.int64, sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0]))
decoder_input_ids = flow.ones(1, 5, 5, dtype=flow.bool, sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0]))
decoder_attn_mask =  flow.ones(1, 3, 3, dtype=flow.bool, sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0]))
encoder_decoder_attn_mask = flow.ones(1, 3, 5, dtype=flow.bool, sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0]))

# output = t5_graph(
#     encoder_input_ids, 
#     encoder_attn_mask, 
#     decoder_input_ids, 
#     decoder_attn_mask, 
#     encoder_decoder_attn_mask
# )
# print(output)

t5_graph._compile(
    encoder_input_ids, 
    encoder_attn_mask, 
    decoder_input_ids, 
    decoder_attn_mask, 
    encoder_decoder_attn_mask
)

convert_to_onnx_and_check(
    t5_graph,
    external_data=False, 
    opset=11, 
    flow_weight_dir=None, 
    onnx_model_path="./", 
    dynamic_batch_size=False,
    device="gpu_global",
)