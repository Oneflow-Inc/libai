# coding=utf-8
import oneflow as flow
from libai.trainer import DefaultTrainer, default_setup
from libai.trainer.trainer import HookBase

# NOTE: Temporarily use yacs as config 
from yacs.config import CfgNode as CN


def setup():
    """
    Create configs and perform basic setups.
    """
    
    cfg = CN()
    cfg.output_dir = "./demo_output"
    cfg.load = None # "./demo_output2/model_0000999"
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

class DemoTrainDataIterHook(HookBase):
    """
    Get training data for model.forward().
    This hook uses the time in the call to its :meth:`before_step` methods.
    """

    def before_step(self):
        # assert self.trainer.train_data_iterator is not None
        # self.trainer._train_data = iter(self.trainer.train_data_iterator)
        data = flow.randn(32, 512).to("cuda")
        if self.trainer.cfg.mode == "graph":
            data = data.to_consistent(sbp=flow.sbp.split(0), placement = flow.env.all_device_placement("cuda"))
        self.trainer._train_data = (data, )

class DemoTrianer(DefaultTrainer):
    def get_train_data_hooks(self):
        return DemoTrainDataIterHook()
    

def main():
    cfg = setup()

    trainer = DemoTrianer(cfg)

    # trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    main()