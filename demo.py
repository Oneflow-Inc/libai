# coding=utf-8

from libai.trainer import DefaultTrainer, default_setup

# NOTE: Temporarily use yacs as config 
from yacs.config import CfgNode as CN


def setup():
    """
    Create configs and perform basic setups.
    """


    cfg = CN()
    cfg.output_dir = "./demo_output2"
    cfg.load = "./demo_output/model_0000999"
    cfg.start_iter = 0
    cfg.train_iters = 6000
    cfg.global_batch_size = 64
    cfg.save_interval = 1000
    cfg.log_interval = 20
    cfg.nccl_fusion_threshold_mb = 16
    cfg.nccl_fusion_max_ops = 24
    cfg.mode = "eager"
    cfg.data_parallel_size = 1
    cfg.micro_batch_size = 32

    default_setup(cfg)
    return cfg

def main():
    cfg = setup()

    trainer = DefaultTrainer(cfg)

    # trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    main()