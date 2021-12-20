import os
from libai.config import LazyConfig, instantiate, default_argument_parser
from libai.models.bert import Bert


def do_train(cfg):
    model = Bert(cfg.model)
    # model = instantiate(cfg.model)
    print(model)

    for i in range(cfg.train.max_iter):
        print(i)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    output_dir = cfg.train.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save config to reproduce the results
    LazyConfig.save(cfg, os.path.join(output_dir, "config.yaml"))

    do_train(cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
