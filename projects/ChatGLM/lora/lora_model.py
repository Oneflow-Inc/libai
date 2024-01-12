# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
# Copyright 2023-present the HuggingFace Inc. team.
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

import logging
import re
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict
from enum import Enum
from itertools import chain
from typing import Any, List, Optional

from oneflow import nn
from tqdm import tqdm

from projects.ChatGLM.lora.layers import BaseTunerLayer, Linear, LoraLayer
from projects.ChatGLM.lora.utils import _get_submodules, check_target_module_exists

logger = logging.getLogger(__name__)


class BaseTuner(nn.Module, ABC):
    r"""
    A base tuner model that provides the common methods and attributes for all
    tuners that are injectable into flow.nn.Module

    """

    def __init__(self, model, peft_config, adapter_name: str) -> None:
        super().__init__()

        # one adapter for begining
        self.model = model
        self.peft_config = {adapter_name: peft_config}
        self.active_adapter = adapter_name
        self.inject_adapter(self.model, adapter_name)

    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    @abstractmethod
    def _check_target_module_exists(peft_config, key: str) -> bool:
        r"""
        A helper private method to check if the passed module's key name matches any of the
        target modules in the `peft_config.target_modules` list. If it does, return `True`,
        else return `False`.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            key (`str`):
                The module's key name.
        """
        ...

    @abstractmethod
    def _create_and_replace(
        self,
        peft_config,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        **optional_kwargs: Any,
    ) -> None:
        r"""
        Inplace replacement of the target module with the adapter layer.
        This method needs to be overriden by all the tuner classes.

        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
            target (`nn.Module`):
                The target module.
            target_name (`str`):
                The target module's name.
            parent (`nn.Module`):
                The parent module.
            **optional_kwargs (`dict`):
                The optional keyword arguments to pass to deal with particular cases
                (e.g. 8bit, 4bit quantization)
        """
        ...

    @abstractmethod
    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        r"""
        A helper method to mark only the adapter layers as trainable
        (i.e. module.requires_grad = False) This needs to be overriden
        for all tuner classes to match the correct key names.

        Check `peft.tuners.lora.LoraModel._mark_only_adapters_as_trainable`
        for an example.
        """
        ...

    def _check_new_adapter_config(self, config) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it
        conflicts with existing adapters.

        """

    def inject_adapter(self, model: nn.Module, adapter_name: str):
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
        """
        peft_config = self.peft_config[adapter_name]
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(peft_config)

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]

        for key in key_list:

            if not self._check_target_module_exists(peft_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(model, key)

            # convert to lora layer
            name_path = key.split(".")
            layer_idx = 0
            for name in name_path:
                if name.isdigit():
                    layer_idx = int(name)
                    break

            if isinstance(target, Linear):
                optional_kwargs = {
                    "current_key": key,
                }
                self._create_and_replace(
                    peft_config, adapter_name, target, target_name, parent, **optional_kwargs
                )
            else:
                target = Linear(
                    target,
                    adapter_name=adapter_name,
                    **self.peft_config[adapter_name],
                    layer_idx=layer_idx,
                )
                setattr(parent, target_name, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

        self._mark_only_adapters_as_trainable(model)

        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

    def merge_adapter(self, safe_merge=False, adapter_names: Optional[list[str]] = None) -> None:
        """
        This method merges the adapter layers into the base model.

        Merging adapters can lead to a speed up of the forward pass. A copy of the adapter
        weights is still kept in memory, which is required to unmerge the adapters. In
        order to merge the adapter weights without keeping them in memory, please call
        `merge_and_unload`.

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original
                weights and check for NaNs before merging the weights. This is useful if
                you want to check if the merge operation will produce NaNs.
                Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active
                adapters will be merged. Defaults to `None`.
        """
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.merge(safe_merge=safe_merge, adapter_names=adapter_names)

    def unmerge_adapter(self):
        """
        This method unmerges all merged adapter layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.unmerge()

    def _unloading_checks(self, adapter_names: Optional[List[str]]):
        adapters_to_consider = adapter_names or self.active_adapters
        is_modules_to_save_available = any(
            self.peft_config[adapter].modules_to_save for adapter in adapters_to_consider
        )
        if is_modules_to_save_available and len(adapters_to_consider) > 1:
            raise ValueError("Cannot unload multiple adapters that specify `modules_to_save`.")


class LoraModel(BaseTuner):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2106.09685.

    Args:
        model : The model to be adapted.
        config : The configuration of the Lora model.
        adapter_name : The name of the adapter, defaults to `"default"`.

    Returns:
        `flow.nn.Module`: The Lora model.

    """

    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it
        conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually
        # has bias != "none",or else the check does not fully correspond to the
        # error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. "
                "When using multiple adapters, set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        return check_target_module_exists(lora_config, key)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(
            chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys())
        )
        target_name_key = next(
            filter(lambda key: re.match(r".*\.{" + key + r"}$", current_key), pattern_keys),
            current_key,
        )

        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }

        kwargs["bias"] = bias

        linear_types = (Linear,)

        if isinstance(target, linear_types):
            target.update_layer(
                adapter_name,
                r,
                alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}'"
                    " does not produce the same output as the the base model would without"
                    " adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn(
                        "Adapter cannot be set when the model is merged. Unmerging the model first."
                    )
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[List[str]] = None,
    ):

        self._unloading_checks(adapter_names)
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)

        return self.model

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[List[str]] = None,
    ) -> nn.Module:
        r"""
        This method merges the LoRa layers into the base model. This is needed if
        someone wants to use the base model as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any
                potential Nan in the adapter weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active
                adapters will be merged. Defaulte to `None`.
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging.
        This gives back the original base model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        raw_state_dict = super().state_dict(destination, prefix, keep_vars)
        lora_state_dict = OrderedDict()
        lora_state_dict._metadata = OrderedDict()
        for key in raw_state_dict:
            if self.prefix in key.lower():
                lora_state_dict[key] = raw_state_dict[key]
        for key in raw_state_dict._metadata:
            lora_state_dict._metadata[key] = raw_state_dict._metadata[key]

        return lora_state_dict
