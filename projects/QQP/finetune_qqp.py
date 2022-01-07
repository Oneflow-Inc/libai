import sys
import oneflow as flow

sys.path.append(".")
from dataset.qqp_dataset import build_train_valid_test_data_iterators

from libai.config import LazyConfig, default_argument_parser
from libai.utils import distributed as dist
from libai.utils.checkpoint import Checkpointer

from libai.trainer import DefaultTrainer, default_setup
import logging

def get_batch(data_iterator):
    """Build the batch."""

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

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
    @classmethod
    def build_train_valid_test_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`libai.data.build_reid_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        logger = logging.getLogger("libai."+__name__)
        logger.info("Prepare training set")
        return build_train_valid_test_data_iterators(cfg)
    

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
