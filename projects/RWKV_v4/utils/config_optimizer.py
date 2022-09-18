def get_RWKV_v4_config_optim(model):
    no_decay = set()

    for mn, m in model.named_modules():  # here we disable weight_decay
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn]
                    for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
   
    return optim_groups