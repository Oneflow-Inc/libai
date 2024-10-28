import oneflow as flow
import oneflow.nn as nn
import torch

from libai.utils.checkpoint import Checkpointer

from .load_detr_weight import convert_state, load_tensor


class detr_checkpointer(Checkpointer):
    def __init__(
        self,
        model: nn.Module,
        save_dir: str = "",
        *,
        save_to_disk: bool = True,
        **checkpointables: object
    ):
        super().__init__(model, save_dir, save_to_disk=save_to_disk, **checkpointables)

    def resume_or_load(self, path: str, *, resume: bool = True, weight_style: str = "pytorch"):
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint (if exists). Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.
        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists.
        Returns:
            same as :meth:`load`.
        """
        assert weight_style == "oneflow" or "pytorch"
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            if weight_style == "oneflow":
                return self.load(path, checkpointables=[])
            if weight_style == "pytorch":
                return self.load_torch_weight(path)

    def load_torch_weight(self, path):

        if not path:
            # no checkpoint provided
            self.logger.info("No pytorch-style checkpoint found. Training model from scratch")
            return {}
        self.logger.info("Loading pytorch-style checkpoint from {}".format(path))

        torch_state_dict = torch.load(path)["model"]
        of_state_dict = convert_state(torch_state_dict)
        for key, value in of_state_dict.items():
            load_tensor(self.model.state_dict()[key], value)
