# Training

To run training, we highly recommend users to use standardized “trainer” in LiBai

## Trainer Abstraction

We provide a standardized “trainer” abstraction with a hook system that helps simplify the standard training behavior.

DefaultTrainer is initialized from lazy config system, used by `tools/train_net.py` and many scripts. It includes many standard default behaviors that one might want to opt in, including default configurations for optimizer, learning rate schedule, logging, evaluation, checkpointing etc.


For simple customizations (e.g. change optimizer, evaluator, LR scheduler, data loader, etc.), modify your `lazycall` method in `config.py` according to your own needs (refer to [Config_System](https://libai.readthedocs.io/en/latest/tutorials/Config_System.html#configs-in-libai)).

## Customize a DefaultTrainer

For complicated customizations, we recommend users to overwrite function in [`DefaultTrainer`](https://github.com/Oneflow-Inc/libai/blob/main/libai/engine/default.py).

In DefaultTrainer, training consists of `run_step in trainer` and `hooks`, you can modify `trainer` and `hooks` according to your own needs

```python
class DefaultTrainer(TrainerBase):
    def train(self, start_iter: int, max_iter: int):
        ...

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train() # in hooks
                for self.iter in range(start_iter, max_iter):
                    self.before_step() # in hooks
                    self.run_step() # in self._trainer
                    self.after_step() # in hooks
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train() # in hooks

```

Refer to `tools/train_net.py`, you can rewrite `tools/my_train_net.py` with your modified `_trainer` and `hooks`. We will introduce how to modify `_trainer` and `hooks` in next subsection.

```python
# tools/my_train_net.py

import ...
from libai.engine import DefaultTrainer
from path_to_myhook import myhook
from path_to_mytraner import _mytrainer

class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # add your _trainer according to your own needs
        # NOTE: run_step() is overwrited in your _trainer
        self._trainer = _mytrainer()

    def build_hooks(self):
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.train.checkpointer.period),
        ]
        # add your hook according to your own needs
        # NOTE: all hooks will be called sequentially 
        ret.append(myhook()) 

        ...

        if dist.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), self.cfg.train.log_period))
        return ret

logger = logging.getLogger("libai." + __name__)

def main(args):
    ...

    trainer = MyTrainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
```

Using a trainer+hook system means there will always be some non-standard behaviors that cannot be supported, especially in research. For this reason, we intentionally keep the trainer & hook system minimal, rather than powerful.

### Add Hooks in Trainer

For extra tasks during training, check the hook system to see if it’s supported.

[`HookBase`](https://github.com/Oneflow-Inc/libai/blob/ffe5ca0e46544d1cbb4fbe88d9185f96c0dc2c95/libai/engine/trainer.py#L28) in `libai/engine/trainer.py` provides a standard behavior for users to use hook. You can overwirte its function according to your own needs. Please refer to [libai/engine/hooks.py](https://github.com/Oneflow-Inc/libai/blob/main/libai/engine/hooks.py)
```python 
class HookBase:
    def before_train(self):
        """
        Called before the first iteration.
        """

    def after_train(self):
        """
        Called after the last iteration.
        """

    def before_step(self):
        """
        Called before each iteration.
        """

    def after_step(self):
        """
        Called after each iteration.
        """
```

As an example, to print `iter` in trainer during training:

```python
class InfoHook(HookBase):
    def before_train(self):
        logger.info(f"start training at {self.trainer.iter}")

    def after_train(self):
        logger.info(f"end training ad {self.trainer.iter}")

    def after_step(self):
        if self.trainer.iter % 100 == 0:
            logger.info(f"iteration {self.trainer.iter}!")
```

Then you can import your `hook` in `tools/my_train_net.py`

### Modify train_step in Trainer

In LiBai, `EagerTrainer` and `GraphTrainer` are supported in `libai/engine/trainer.py`. `EagerTrainer` supports `train_step` for `eager` mode while `GraphTrainer` supports `train_step` for `graph` mode, the mode is depended on `graph.enabled` in your `config.py`.

> more details about `eager` and `graph` mode, please refer to [oneflow doc](https://docs.oneflow.org/en/master/basics/08_nn_graph.html)

As an example, to add a temp variable for model output in run_step

```python
class MyEagerTrainer(EagerTrainer):

    def __init__(self, model, data_loader, optimizer, grad_acc_steps=1):
        super().__init__(model, data_loader, optimizer, grad_acc_steps)
        self.previous_output = None

    def run_step(self, get_batch: Callable):
        ...
        loss_dict = self.model(**data)
        self.previous_output = loss_dict
        ...
```

Then you can set your `MyEagerTrainer` as `self.trainer` in `tools/my_train_net.py`

## Logging of Metrics

During training, trainer put metrics to a centralized [EventStorage](https://libai.readthedocs.io/en/latest/modules/libai.utils.html#module-libai.utils.events). You can use the following code to access it and log metrics to it:

```python
from libai.utils.events import get_event_storage

# inside the model:
if self.training:
  value = # compute the value from inputs
  storage = get_event_storage()
  storage.put_scalar("some_accuracy", value)

```

See [EventStorage](https://libai.readthedocs.io/en/latest/modules/libai.utils.html#module-libai.utils.events) for more details.

Metrics are then written to various destinations with EventWriter. Metrics information will be written to `{cfg.train.output_dir}/metrics.json`. DefaultTrainer enables a few EventWriter with default configurations. See above for how to customize them.