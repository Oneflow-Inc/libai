from libai.engine import DefaultTrainer
import oneflow as flow


def sync():
    flow.comm.barrier()


class RematTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def after_step(self):
        sync()
        super().after_step()
