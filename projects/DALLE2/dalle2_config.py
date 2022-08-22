from libai.config import LazyCall
from dalle2_model_config import dalle2_model as model
from configs.common.train import train
from dalle2.tokenizer import SimpleTokenizer

model.prior_weight_path = "./dalle2/model_weights/prior_aes_finetune.pth"
model.decoder_weight_path = "./dalle2/model_weights/latest.pth"

tokenizer = LazyCall(SimpleTokenizer)()