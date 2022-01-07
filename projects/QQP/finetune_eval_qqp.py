from functools import total_ordering
import sys
import numpy as np
import oneflow as flow

sys.path.append(".")
from dataset.qqp_dataset import build_train_valid_test_data_iterators
from evaluation.eval_hook import EvalHook

from libai.config import LazyConfig, default_argument_parser
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer

from libai.trainer import DefaultTrainer, default_setup, hooks
import logging

logger = logging.getLogger("libai."+__name__)

def pad_batch(x_list, batch_size):
    x = x_list[0]
    valid_sample = x.shape[0]
    assert valid_sample <= batch_size
    # check all batch size is equal
    for xi in x_list[1:]:
        assert xi.shape[0] == valid_sample
    
    if valid_sample == batch_size:
        return x_list, batch_size
    # pad all data
    padded_list = []
    for xi in x_list: 
        pad_shape = (batch_size, *xi.shape[1:])
        padded_xi = flow.zeros(pad_shape, sbp=xi.sbp, placement=xi.placement, dtype=xi.dtype)
        padded_xi[:valid_sample, ...] = padded_xi[:valid_sample, ...] + xi
        padded_xi[valid_sample:, ...] = padded_xi[valid_sample:, ...] + xi[0].unsqueeze(0)
        padded_list.append(padded_xi)
    return padded_list, valid_sample    


def get_batch(data_iterator, data=None):
    """Build the batch."""

    if data_iterator is not None:
        data = next(data_iterator)

    input_placement = dist.get_layer_placement(0)
    label_placement = dist.get_layer_placement(-1)
    sbp = dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast])

    def to_consistent(tensor, placement):
        tensor = tensor.to_consistent(placement, sbp)
        return tensor

    # Unpack.
    tokens = to_consistent(data["text"].long(), input_placement)
    types = to_consistent(data["types"].long(), input_placement)
    padding_mask = to_consistent(data["padding_mask"].long(), input_placement)
    label = to_consistent(data["label"].long(), label_placement)

    return tokens, padding_mask, types, label

class Trainer(DefaultTrainer):
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PeriodicCheckpointer(
                self.checkpointer, self.cfg.train.checkpointer.period
            ),
        ]
        
        def test_and_save_results():
            if not hasattr(self, "graph_eval"):
                self.graph_eval = self.build_graph(self.cfg, self.model, is_train=False)
            self._last_eval_results = self.test(self.cfg, self.graph_eval)
            return self._last_eval_results

        # Do evaluation before checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(EvalHook(self.cfg.train.eval_period, test_and_save_results))
        
        if dist.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(
                hooks.PeriodicWriter(self.build_writers(), self.cfg.train.log_period)
            )
        return ret
    
    @classmethod
    def build_train_valid_test_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`libai.data.build_reid_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        
        logger.info("Prepare training set")
        return build_train_valid_test_data_iterators(cfg)
    
    def test(self, cfg, graph_eval):
        # valid_data_iterator is valid_dataloader
        total_sample = len(self.valid_data_iterator)
        top1_num = flow.zeros(1, dtype=flow.float32)
        num_sample = 0
        for idx, data in enumerate(self.valid_data_iterator):
            tokens, padding_mask, types, label = get_batch(None, data)
            [tokens, padding_mask, types, label], valid_sample = pad_batch(
                [tokens, padding_mask, types, label],
                batch_size=cfg.train.global_batch_size
            )
            pred = graph_eval(tokens, padding_mask, types)
            pred = pred[:valid_sample, ...]
            label = label[:valid_sample, ...]
            clsidxs = pred.argmax(dim=-1)
            clsidxs = clsidxs.to(flow.int32)
            match = (clsidxs == label).sum()
            top1_num += match.item()
            num_sample += valid_sample
        logger.info(f"acc: {top1_num.item()/num_sample}, {top1_num.item()}/{num_sample, total_sample}")

    def run_step(self):
        return super().run_step(get_batch)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        graph = Trainer.build_graph(cfg, model, is_train=False)
        res = Trainer.test(cfg, graph)

    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
