import oneflow as flow

from utils.weight_convert import load_torch_checkpoint, load_torch_checkpoint_linear    


def build_model(finetune, model):
    linear_keyword = "head"
    for name, param in model.named_parameters():
        if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
            param.requires_grad = False

    assert finetune.weight_style in ["oneflow", "pytorch"]
    if finetune.enable == True:
        if finetune.weight_style == "oneflow":
                model.load_state_dict(flow.load(finetune.finetune_path))
        else:
            model = load_torch_checkpoint(model, path=finetune.finetune_path, strict=False, linear_keyword=linear_keyword)
            # model = load_torch_checkpoint_linear(model, path=finetune.inference_path, strict=False)
        getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
        getattr(model, linear_keyword).bias.data.zeros_()

    return model