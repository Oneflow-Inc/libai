import oneflow as flow
import sys
sys.path.append(".")

from libai.config import try_get_key
from utils.weight_convert import load_torch_checkpoint, load_torch_checkpoint_linear    
from libai.engine import DefaultTrainer


class VitFinetuningTrainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        if try_get_key(cfg, "finetune") is not None:
            linear_keyword = "head"
            for name, param in model.named_parameters():
                if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                    param.requires_grad = False

            assert cfg.finetune.weight_style in ["oneflow", "pytorch"]
            if cfg.finetune.enable == True:
                if cfg.finetune.weight_style == "oneflow":
                    model.load_state_dict(flow.load(cfg.finetune.finetune_path))
                else:
                    model = load_torch_checkpoint(model, path=cfg.finetune.finetune_path, strict=False, linear_keyword=linear_keyword)
                    # model = load_torch_checkpoint_linear(model, path=cfg.finetune.inference_path, strict=False)
                getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
                getattr(model, linear_keyword).bias.data.zeros_()

        return model