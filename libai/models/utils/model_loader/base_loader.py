# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import copy
import logging
import os

import omegaconf
import oneflow as flow
from termcolor import colored

import libai.utils.distributed as dist
from libai.config import LazyCall
from libai.models.build import build_model

logger = logging.getLogger(__name__)


WEIGHTS_NAME_PT = "pytorch_model.bin"
CONFIG_NAME = "config.json"


def _load_state_dict_into_model(model_to_load, state_dict, start_prefix):
    """load state dict into model

    Args:
        model_to_load (nn.Module): Model to be loaded.
        state_dict (OrderedDict): State dict of pretrained model.
        start_prefix (str): Start prefix.

    Returns:
        list: error message about loading.
    """
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model_to_load, prefix=start_prefix)

    return error_msgs


class ModelLoader(object):
    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        """Class used to load the [`transformers`](https://huggingface.co/models) pretrained model
        or `OneFlow` pretrained model.

        Args:
            model (libai.models): Model to be loaded in Libai.
            libai_cfg (dict): The config of model in LiBai, you can import it from
                `libai.config.configs.common.models`.
            pretrained_model_path (str): The directory path of pretrained model,
                which contains model weights file and config file.
            output_loading_info (`bool`, *optional*, defaults to `False`):
                Whether to return a dictionary containing missing keys, unexpected keys
                and error messages.
        """
        self.model = model
        self.libai_cfg = libai_cfg
        self.pretrained_model_path = pretrained_model_path
        self.kwargs = kwargs
        self.output_loading_info = kwargs.pop("output_loading_info", False)

    def _state_dict_to_global(self, flow_state_dict=None, mode="libai"):
        """Tensor in OneFlow state dict to global according to model's sbp and placement.

        Args:
            flow_state_dict (OrderedDict): State dict of OneFlow's pretrained model.
        """
        assert mode in ["libai", "pytorch"], f"not support for mode {mode}"
        if mode == "libai" or dist.is_main_process():
            prefix = self.base_model_prefix_2

            # Checkpoint
            has_prefix_module = any(
                s.startswith(self.base_model_prefix_2) for s in flow_state_dict.keys()
            )
            # Module
            expects_prefix_module = any(
                s.startswith(prefix) for s in self.model.state_dict().keys()
            )

            start_prefix = "" if has_prefix_module else prefix + "."
            loaded_keys = [start_prefix + key for key in flow_state_dict.keys()]
        else:
            prefix, has_prefix_module, expects_prefix_module, loaded_keys = [None] * 4
            flow_state_dict = collections.OrderedDict()

        prefix = dist.broadcast_py_object(prefix, src=0)
        has_prefix_module = dist.broadcast_py_object(has_prefix_module, src=0)
        expects_prefix_module = dist.broadcast_py_object(expects_prefix_module, src=0)
        loaded_keys = dist.broadcast_py_object(loaded_keys, src=0)

        # to global
        for key, value in self.model.state_dict().items():
            if not expects_prefix_module:
                key = prefix + "." + key
            if key in loaded_keys:
                if not has_prefix_module:
                    key = ".".join(key.split(".")[1:])

                if mode == "pytorch":
                    flow_state_dict[key] = flow.to_global(
                        flow_state_dict[key] if dist.is_main_process() else flow.Tensor(None),
                        sbp=flow.sbp.broadcast,
                        placement=flow.placement("cpu", ranks=[0]),
                    )

                flow_state_dict[key] = flow.to_global(
                    flow_state_dict[key],
                    sbp=value.sbp,
                    placement=flow.placement("cpu", ranks=list(value.placement.ranks)),
                )
        return flow_state_dict

    def _load_pretrained_model(
        self,
        model,
        state_dict,
        pretrained_model_path,
        ignore_mismatched_sizes=False,
    ):
        """Load pretrained model.

        Args:
            model (libai.models): The model to be loaded.
            state_dict (OrderedDict): state dict.
            loaded_keys (list): keys of state dict.
            pretrained_model_path (str): pretrained modelE path.
            ignore_mismatched_sizes (bool):
                Whether or not to raise an error if some of the weights
                from the checkpoint do not have the same size as the
                weights of the model, defaults to `False`.
        """
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        prefix = self.base_model_prefix_2

        loaded_keys = state_dict.keys()
        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(prefix)]
            expected_keys = [
                ".".join(s.split(".")[1:]) if s.startswith(prefix) else s for s in expected_keys
            ]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        start_prefix = ""
        model_to_load = model
        if (
            len(self.base_model_prefix_2) > 0
            and not hasattr(model, self.base_model_prefix_2)
            and has_prefix_module
        ):
            start_prefix = self.base_model_prefix_2 + "."
        if (
            len(self.base_model_prefix_2) > 0
            and hasattr(model, self.base_model_prefix_2)
            and not has_prefix_module
        ):
            model_to_load = getattr(model, self.base_model_prefix_2)
            if any(key in expected_keys_not_prefixed for key in loaded_keys):
                raise ValueError("The state dict of the model you are loading is corrupted.")

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (
                                checkpoint_key,
                                state_dict[checkpoint_key].shape,
                                model_state_dict[model_key].shape,
                            )
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict, start_prefix)

        if dist.get_local_rank() == 0:
            if len(error_msgs) > 0:
                error_msg = "\n\t".join(error_msgs)
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}"
                )
            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_path} "
                    "were not used when "
                    f"initializing {model.__class__.__name__}:\n {unexpected_keys}\n"
                )
            else:
                logger.info(
                    f"All model checkpoint weights were used when initializing "
                    f"{model.__class__.__name__}.\n"
                )
            if len(missing_keys) > 0:
                logger.warning(
                    f"Some weights of {model.__class__.__name__} were not initialized "
                    f"from the model checkpoint at {pretrained_model_path}:\n "
                    f"{missing_keys} \n"
                )
            elif len(mismatched_keys) == 0:
                logger.info(
                    f"All the weights of {model.__class__.__name__} were initialized "
                    f"from the model checkpoint at {pretrained_model_path}.\n"
                )
            if len(mismatched_keys) > 0:
                mismatched_warning = "\n".join(
                    [
                        f"- {key}: found shape {shape1} in the checkpoint and {shape2}"
                        "in the model instantiated"
                        for key, shape1, shape2 in mismatched_keys
                    ]
                )
                logger.warning(
                    f"Some weights of {model.__class__.__name__} were not initialized"
                    f"from the model checkpoint at {pretrained_model_path} "
                    f"and are newly initialized because the shapes did not"
                    f"match:\n{mismatched_warning}\n"
                )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs


class ModelLoaderLiBai(ModelLoader):
    """Class used to load `OneFlow` pretrained model.

    Args:
        model (libai.models): Model to be loaded in Libai.
        libai_cfg (dict): The config of model in LiBai, you can import it from
            `libai.config.configs.common.models`.
        pretrained_model_path (str): The file path of pretrained model.
        output_loading_info (`bool`, *optional*, defaults to `False`):
            Whether to return a dictionary containing missing keys, unexpected keys
            and error messages.
    """

    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_2 = None  # prefix in LiBai

    def _load_flow_state_dict(self, state_dict_file):
        # load oneflow_model
        state_dict = flow.load(state_dict_file, global_src_rank=0)
        return state_dict

    def load(self):
        """Load model.

        # For example:

        # .. code-block:: python

            >>> import libai
            >>> from libai.config.configs.common.models.bert import cfg
            >>> from model_loader import BertLoaderLiBai

            >>> loder = BertLoaderLiBai(
                    libai.models.BertModel,
                    cfg,
                    'path/bert-base-chinese'
                )
            >>> bert = loder.load()

        """

        flow_state_dict = self._load_flow_state_dict(self.pretrained_model_path)

        # Instance model
        if isinstance(self.model, omegaconf.dictconfig.DictConfig):
            self.model.cfg = self.libai_cfg
            self.model = build_model(self.model)
        else:
            self.model = build_model(LazyCall(self.model)(cfg=self.libai_cfg))

        # State_dict to global
        self._state_dict_to_global(flow_state_dict, mode="libai")

        # Load
        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            error_msgs,
        ) = self._load_pretrained_model(self.model, flow_state_dict, self.pretrained_model_path)

        if self.output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info
        return model


class ModelLoaderHuggerFace(ModelLoader):
    """Class used to load the [`transformers`](https://huggingface.co/models)
    pretrained model.
    """

    def __init__(self, model, libai_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, libai_cfg, pretrained_model_path, **kwargs)
        self.base_model_prefix_1 = None  # prefix in Transformers
        self.base_model_prefix_2 = None  # prefix in LiBai
        self.origin_libai_cfg = copy.deepcopy(self.libai_cfg)
        self.changed_keys = set()  # Store the changed configuration

    def _convert_tensor(self, tensor):
        """Convert PyTorch tensor to OneFlow tensor.

        Args:
            tensor (torch.Tensor): The source tensor.

        Returns:
            flow.Tensor: The target tensor.
        """
        return flow.Tensor(tensor.detach().cpu().numpy())

    def _convert_tensors(self, torch_state_dict):

        for k, v in torch_state_dict.items():
            torch_state_dict[k] = self._convert_tensor(v)

        return torch_state_dict

    def _fix_key(self, state_dict):
        """Fix the key in state dict: Convert "gamma" to "weight" and "beta" to "bias".

        Args:
            state_dict (OrderedDict): state dict of pretrained model.

        Returns:
            OrderedDict: State dict after fix key.
        """
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        return state_dict

    def _fix_qkv_ordering(
        self, qkv, head_size, num_heads, hidden_size=None, checkpoint_version=0.0
    ):
        # TODO(xzp): Different versions checkpoint

        hidden_size = (head_size * num_heads) if hidden_size is None else hidden_size
        num_of_qkv = qkv.shape[0] // (head_size * num_heads)
        mode = "weight" if qkv.ndim > 1 else "bias"
        if mode == "weight":
            qkv = qkv.view([num_of_qkv, num_heads, head_size, hidden_size])
            qkv = (
                qkv.permute(1, 0, 2, 3)
                .contiguous()
                .view(num_of_qkv * head_size * num_heads, hidden_size)
            )
        elif mode == "bias":
            qkv = qkv.view(num_of_qkv, num_heads, head_size)
            qkv = qkv.permute(1, 0, 2).contiguous().view(-1)
        return qkv

    def _convert_state_dict(self, flow_state_dict, cfg):
        """A function used to convert the checkpoint file of Huggingface to LiBai.

        Args:
            torch_state_dict (OrderedDict): torch state dict.
            cfg (dict): model's default config dict in LiBai.

        Returns:
            OrderedDict: flow state dict.
        """
        raise NotImplementedError("_convert_state_dict not implemented")

    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.

        Args:
            config_file (str): Path of config file.
        """

        raise NotImplementedError("_load_config_from_json not implemented")

    def _load_torch_state_dict(self, state_dict_file):
        try:
            import torch
        except ImportError:
            raise ImportError("Load torch state dict need torch.")

        # load pytorch_model.bin
        if isinstance(state_dict_file, str):
            return torch.load(state_dict_file, map_location="cpu")

        if isinstance(state_dict_file, list):
            merged_state_dict = {}
            for file in state_dict_file:
                state_dict = torch.load(file, map_location="cpu")
                merged_state_dict.update(state_dict)
            return merged_state_dict

    def _update_cfg(self, keys_libai, value_target):
        """Update the libai_cfg according to target_cfg.

        Args:
            keys_libai (str): The key of libai_cfg.
            value_target (int | float): The value of target_cfg.
        """
        if keys_libai not in self.libai_cfg.keys():
            return
        if self.libai_cfg[keys_libai] != value_target:
            self.libai_cfg[keys_libai] = value_target

    def _update_cfg_log(self):
        if dist.get_local_rank() == 0:
            for key in sorted(self.libai_cfg):
                if self.origin_libai_cfg[key] == self.libai_cfg[key]:
                    continue
                self.changed_keys.add(key)
                temp_key = colored(key, "yellow")
                logger.info(
                    f"changed libai model cfg {temp_key} : "
                    f"{self.origin_libai_cfg[key]} -> {self.libai_cfg[key]} "
                )
            if len(self.changed_keys) > 0:
                logger.warning(
                    "The following model configurations has been modified according "
                    "to `config.json` or kwargs: \n"
                    f"{self.changed_keys} \n"
                )

            if dist.get_pipeline_parallel_size() > 1:
                logger.warning(
                    colored(
                        "If you use pipeline parallel, please "
                        "confirm the setting of `train.dist.pipeline_num_layers` \n",
                        "red",
                    )
                )

    def load(self):
        """Load model.

        # For example:

        # .. code-block:: python

            >>> import libai
            >>> from configs.common.models.bert import cfg
            >>> from libai.models.utils import BertLoaderHuggerFace

            >>> loader = BertLoaderHuggerFace(
                    libai.models.BertModel,
                    cfg,
                    'path/bert-base-chinese'
                )
            >>> bert = loader.load()

        """
        if dist.is_main_process():
            if os.path.isdir(self.pretrained_model_path):
                # state_dict file pytorch
                model_files = [
                    os.path.join(self.pretrained_model_path, file)
                    for file in os.listdir(self.pretrained_model_path)
                    if file.endswith(".bin")
                ]

                if len(model_files) == 0:
                    raise EnvironmentError(
                        f"Error: no file named endswith '.bin' found"
                        f"in directory {self.pretrained_model_path}."
                    )

                # config file
                if os.path.isfile(os.path.join(self.pretrained_model_path, CONFIG_NAME)):
                    config_file = os.path.join(self.pretrained_model_path, CONFIG_NAME)

                    # Load config and update config.
                    self._load_config_from_json(config_file)
                else:
                    import warnings

                    warnings.warn(
                        f"Error no file named {CONFIG_NAME} found in directory"
                        f"{self.pretrained_model_path}",
                        RuntimeWarning,
                    )
            else:
                raise EnvironmentError(f"{self.pretrained_model_path} is not a directory.")

            logger.info("loading torch model...")
            torch_state_dict = self._load_torch_state_dict(model_files)
            torch_state_dict = self._fix_key(torch_state_dict)
            logger.info("transfering torch model into oneflow model...")
            flow_state_dict = self._convert_tensors(torch_state_dict)
            flow_state_dict = self._convert_state_dict(torch_state_dict, self.libai_cfg)
        else:
            flow_state_dict = None

        self.libai_cfg = dist.broadcast_py_object(self.libai_cfg, src=0)

        # Instance model
        logger.info("building LiBai model...")
        if isinstance(self.model, omegaconf.dictconfig.DictConfig):
            self.model.cfg = self.libai_cfg
            self.model = build_model(self.model)
        else:
            self.model = build_model(LazyCall(self.model)(cfg=self.libai_cfg))

        # State_dict to global
        logger.info("transfering state_dict local to global...")
        flow_state_dict = self._state_dict_to_global(flow_state_dict, mode="pytorch")

        logger.info("loading model weights into LiBai...")
        # Load
        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            error_msgs,
        ) = self._load_pretrained_model(self.model, flow_state_dict, self.pretrained_model_path)

        if self.output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info
        return model
