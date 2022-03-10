import time
from typing import Callable

from libai.trainer import DefaultTrainer
from libai.trainer.trainer import EagerTrainer



class MoCoDefaultTrainer(DefaultTrainer):

    def __init__(self, cfg):

        super().__init__(cfg)

        self.model.max_iter = cfg.train.train_epoch

        self._trainer = MoCoEagerTrainer(
                self.model, self.train_loader, self.optimizer, cfg.train.num_accumulation_steps
            )


class MoCoEagerTrainer(EagerTrainer):

    def __init__(self, model, data_loader, optimizer, grad_acc_steps=1):

        super().__init__(model, data_loader, optimizer, grad_acc_steps)

        self.m = self.model.m

    def run_step(self, get_batch: Callable):

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        # If you want to do something with the data, you can wrap the dataloader.
        data = next(self._data_loader_iter)
        data = get_batch(data, getattr(self.data_loader, "mixup_func", None))
        data_time = time.perf_counter() - start

        # update the moco_momentum per step
        loss_dict, m_dict = self.model(**data, cu_iter=self.iter, m=self.m)
        self.m = m_dict["m"]

        losses = sum(loss_dict.values()) / self.grad_acc_steps
        losses.backward()
        self.write_metrics(loss_dict, data_time)

        if (self.iter + 1) % self.grad_acc_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
