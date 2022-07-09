import json
import logging
import os

import oneflow as flow
import torch
from yaml import warnings

from libai.config import LazyCall
from libai.models import build_model

logger = logging.getLogger(__name__)


WEIGHTS_NAME_PT = "pytorch_model.bin"
WEIGHTS_NAME_OF = "oneflow_model"
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


class LoadPretrainedBase(object):
    def __init__(self, model, default_cfg, pretrained_model_path, **kwargs):
        """Class used to load the [`transformers`](https://huggingface.co/models) pretrained model
        or `OneFlow` pretrained model.

        Args:
            model (libai.models): Model to be loaded in Libai.
            default_cfg (dict): The default config of model, you can import it from
                `libai.config.configs.common.models`.
            pretrained_model_path (str): The directory path of pretrained model,
                which contains model weights file and config file.
            output_loading_info (`bool`, *optional*, defaults to `False`):
                Whether to return a dictionary containing missing keys, unexpected keys
                and error messages.
        """
        self.model = model
        self.default_cfg = default_cfg
        self.pretrained_model_path = pretrained_model_path
        self.kwargs = kwargs
        self.output_loading_info = kwargs.pop("output_loading_info", False)
        self.base_model_prefix_1 = None
        self.base_model_prefix_2 = None

    def convert_tensor(self, tensor):
        """Convert pytorch tensor to OneFlow tensor.

        Args:
            tensor (torch.Tensor): The source tensor.

        Returns:
            flow.Tensor: The target tensor.
        """
        tensor = tensor.float()
        return flow.Tensor(tensor.cpu().numpy())

    def _state_dict_to_global(self, flow_state_dict):
        """Tensor in OneFlow state dict to global according to model's sbp and placement.

        Args:
            flow_state_dict (OrderedDict): State dict of OneFlow's pretrained model.
        """
        prefix = self.base_model_prefix_2

        # Checkpoint
        has_prefix_module = any(
            s.startswith(self.base_model_prefix_2) for s in flow_state_dict.keys()
        )
        # Module
        expects_prefix_module = any(s.startswith(prefix) for s in self.model.state_dict().keys())

        start_prefix = "" if has_prefix_module else prefix + "."
        loaded_keys = [start_prefix + key for key in flow_state_dict.keys()]

        # to global
        for key, value in self.model.state_dict().items():
            if not expects_prefix_module:
                key = prefix + "." + key
            if key in loaded_keys:
                if not has_prefix_module:
                    key = ".".join(key.split(".")[1:])
                flow_state_dict[key] = flow.to_global(
                    flow_state_dict[key], sbp=value.sbp, placement=value.placement
                )

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

    def _fix_qkv_ordering(self, qkv, head_size, num_heads, checkpoint_version=0.0):
        # TODO(xzp): Different versions checkpoint

        hidden_size = head_size * num_heads
        num_of_qkv = qkv.shape[0] // hidden_size
        mode = "weight" if qkv.ndim > 1 else "bias"
        if mode == "weight":
            qkv = qkv.view([num_of_qkv, num_heads, head_size, hidden_size])
            qkv = qkv.permute(1, 0, 2, 3).contiguous().view(num_of_qkv * hidden_size, hidden_size)
        elif mode == "bias":
            qkv = qkv.view(num_of_qkv, num_heads, head_size)
            qkv = qkv.permute(1, 0, 2).contiguous().view(-1)
        return qkv

    def _convert_state_dict(self, torch_state_dict, cfg):
        """A function used to convert the checkpoint file of Huggingface to LiBai.

        Args:
            torch_state_dict (OrderedDict): torch state dict.
            cfg (dict): model's default config dict.

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
        # load pytorch_model.bin
        state_dict = torch.load(state_dict_file, map_location="cpu")
        return state_dict

    def _load_flow_state_dict(self, state_dict_file):
        # load oneflow_model
        state_dict = flow.load(state_dict_file)
        return state_dict

    def _load_pretrained_model(
        self,
        model,
        state_dict,
        loaded_keys,
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
                raise ValueError(
                    "The state dictionary of the model you are training to load is corrupted. \
                    Are you sure it was "
                    "properly saved?"
                )

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

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            raise RuntimeError(
                f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}"
            )

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_path} "
                "were not used when "
                f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                f"- This IS expected if you are initializing {model.__class__.__name__} "
                "from the checkpoint of a model trained on another task "
                f"or with another architecture (e.g. initializing a BertForSequenceClassification "
                "model from a BertForPreTraining model).\n"
                f"- This IS NOT expected if you are initializing {model.__class__.__name__} "
                "from the checkpoint of a model that you expect "
                f"to be exactly identical (initializing a BertForSequenceClassification model "
                "from a BertForSequenceClassification model)."
            )
        else:
            logger.info(
                f"All model checkpoint weights were used when initializing "
                f"{model.__class__.__name__}.\n"
            )
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized "
                f"from the model checkpoint at {pretrained_model_path} "
                f"and are newly initialized: {missing_keys}\n"
                f"You should probably TRAIN this model on a down-stream task to be"
                "able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized "
                f"from the model checkpoint at {pretrained_model_path}.\n"
                f"If your task is similar to the task the model of the checkpoint "
                "was trained on, "
                f"you can already use {model.__class__.__name__} for predictions "
                "without further training."
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
                f"You should probably TRAIN this model on a down-stream"
                "task to be able to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    def load_model(self):
        """Load model.

        # For example:

        # .. code-block:: python

            >>> import libai
            >>> from libai.config.configs.common.models.bert import cfg
            >>> from model_utils import LoadPretrainedModels

            >>> my_class = LoadPretrainedBert(
                    libai.models.BertModel,
                    cfg,
                    'path/bert-base-chinese'
                )
            >>> bert = my_class.load_model()

        """
        if os.path.isdir(self.pretrained_model_path):
            # state_dict file pytorch
            if os.path.isfile(os.path.join(self.pretrained_model_path, WEIGHTS_NAME_PT)):
                self.mode = "pt"
                model_file = os.path.join(self.pretrained_model_path, WEIGHTS_NAME_PT)

            # state_dict file oneflow
            elif os.path.isdir(os.path.join(self.pretrained_model_path, WEIGHTS_NAME_OF)):
                self.mode = "of"
                model_file = os.path.join(self.pretrained_model_path, WEIGHTS_NAME_OF)

            else:
                raise EnvironmentError(
                    f"Error no file named {WEIGHTS_NAME_PT} or {WEIGHTS_NAME_OF} found"
                    f"in directory {self.pretrained_model_path}."
                )

            # config file
            if os.path.isfile(os.path.join(self.pretrained_model_path, CONFIG_NAME)):
                config_file = os.path.join(self.pretrained_model_path, CONFIG_NAME)

                # Load config and update config.
                self._load_config_from_json(config_file)
            else:
                warnings.warn(
                    f"Error no file named {CONFIG_NAME} found in directory"
                    f"{self.pretrained_model_path}",
                    RuntimeWarning,
                )
        else:
            raise EnvironmentError(f"{self.pretrained_model_path} is not a directory.")

        if self.mode == "pt":
            torch_state_dict = self._load_torch_state_dict(model_file)
            torch_state_dict = self._fix_key(torch_state_dict)
            flow_state_dict = self._convert_state_dict(torch_state_dict, self.default_cfg)
        else:
            flow_state_dict = self._load_flow_state_dict(model_file)

        loaded_state_dict_keys = list(flow_state_dict.keys())

        # Instance model
        self.model = build_model(LazyCall(self.model)(cfg=self.default_cfg))

        # State_dict to global
        self._state_dict_to_global(flow_state_dict)

        # Load
        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            error_msgs,
        ) = self._load_pretrained_model(
            self.model, flow_state_dict, loaded_state_dict_keys, self.pretrained_model_path
        )

        model.eval()

        if self.output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info
        return model


class LoadPretrainedBert(LoadPretrainedBase):
    def __init__(self, model, default_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, default_cfg, pretrained_model_path, **kwargs)
        
        """NOTE: base_model_prefix_1 is BERT's prefix in Transformers.
        base_model_prefix_2 is BERT's prefix in LiBai."""
        self.base_model_prefix_1 = "bert"
        self.base_model_prefix_2 = "bert"

    def _convert_state_dict(self, torch_state_dict, cfg):
        """Convert torch state dict to flow state dict.

        Args:
            torch_state_dict (OrderedDict): torch state dict.
            cfg (dict): model's default config dict.

        Returns:
            OrderedDict: flow state dict.
        """
        # The converted checkpoint.
        oneflow_state_dict = torch_state_dict.copy()

        # Get configs
        num_heads = cfg.get("num_attention_heads")
        hidden_size = cfg.get("hidden_size")
        layers = cfg.get("hidden_layers")
        head_size = int(hidden_size / num_heads)

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)

        prefix = "bert." if has_prefix else ""
        index_idx = 3 if has_prefix else 2
        qkv_idx = 6 if has_prefix else 5

        old_keys = oneflow_state_dict.keys()

        for key in list(old_keys):

            # Convert bert's embedding layers
            if "embeddings" in key:
                if "word_embeddings" in key:
                    new_key = key.replace("word_embeddings", "vocab_embeddings")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "token_type_embeddings" in key:
                    new_key = key.replace("token_type_embeddings", "tokentype_embeddings")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "LayerNorm.weight" in key:
                    new_key = prefix + "encoders.0.input_layernorm.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "LayerNorm.bias" in key:
                    new_key = prefix + "encoders.0.input_layernorm.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                else:
                    oneflow_state_dict[key] = self.convert_tensor(oneflow_state_dict[key])

            # Convert bert's attention layers
            elif "attention" in key:
                if "self" in key:
                    index = key.split(".")[index_idx]
                    if (
                        prefix + "encoders." + index + ".self_attention.query_key_value.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    q_w = key.replace(key.split(".")[qkv_idx], "query").replace(
                        key.split(".")[qkv_idx + 1], "weight"
                    )
                    k_w = q_w.replace("query", "key")
                    v_w = q_w.replace("query", "value")
                    q_b = q_w.replace("weight", "bias")
                    k_b = k_w.replace("weight", "bias")
                    v_b = v_w.replace("weight", "bias")

                    qkv_w = torch.cat(
                        (
                            oneflow_state_dict.pop(q_w),
                            oneflow_state_dict.pop(k_w),
                            oneflow_state_dict.pop(v_w),
                        ),
                        dim=0,
                    )
                    qkv_b = torch.cat(
                        (
                            oneflow_state_dict.pop(q_b),
                            oneflow_state_dict.pop(k_b),
                            oneflow_state_dict.pop(v_b),
                        ),
                        dim=-1,
                    )

                    qkv_w = self._fix_qkv_ordering(qkv_w, head_size, num_heads)
                    qkv_b = self._fix_qkv_ordering(qkv_b, head_size, num_heads)

                    new_key = (
                        prefix + "encoders." + index + ".self_attention.query_key_value.weight"
                    )
                    oneflow_state_dict[new_key] = self.convert_tensor(qkv_w)

                    new_key = prefix + "encoders." + index + ".self_attention.query_key_value.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(qkv_b)
                elif "output" in key:
                    index = key.split(".")[index_idx]
                    if "dense" in key:
                        if "weight" in key:
                            new_key = prefix + "encoders." + index + ".self_attention.dense.weight"
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                        elif "bias" in key:
                            new_key = prefix + "encoders." + index + ".self_attention.dense.bias"
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                    elif "LayerNorm" in key:
                        if "weight" in key:
                            new_key = (
                                prefix + "encoders." + index + ".post_attention_layernorm.weight"
                            )
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                        elif "bias" in key:
                            new_key = (
                                prefix + "encoders." + index + ".post_attention_layernorm.bias"
                            )
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )

            # Convert bert's intermediate layers
            elif "intermediate" in key:
                index = key.split(".")[index_idx]
                if (
                    prefix + "encoders." + index + ".mlp.dense_h_to_4h.weight"
                    in oneflow_state_dict.keys()
                ):
                    continue
                if "weight" in key:
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = prefix + "encoders." + index + ".mlp.dense_h_to_4h.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(w))
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(b))

            # Convert bert's output layers
            elif "output" in key:
                index = key.split(".")[index_idx]
                if "dense.weight" in key:
                    if (
                        prefix + "encoders." + index + ".mlp.dense_4h_to_h.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    new_key = prefix + "encoders." + index + ".mlp.dense_4h_to_h.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(w))
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(b))
                elif "LayerNorm.weight" in key:
                    if (
                        prefix + "encoders." + str(int(index) + 1) + ".input_layernorm.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    if index == str(layers - 1):
                        new_key = prefix + "final_layernorm.weight"
                        oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(w))
                        new_key = new_key.replace("weight", "bias")
                        oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(b))
                        continue
                    new_key = prefix + "encoders." + str(int(index) + 1) + ".input_layernorm.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(w))
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(b))

            # Convert bert's pooler layers
            elif "pooler" in key:
                if "weight" in key:
                    new_key = prefix + "pooler.dense.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "bias" in key:
                    new_key = prefix + "pooler.dense.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))

            # Convert cls_head layers
            elif "cls" in key:
                if "predictions.bias" in key:
                    new_key = "cls_head.lm_logits.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "dense.weight" in key:
                    new_key = "cls_head.predictions.dense.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "dense.bias" in key:
                    new_key = "cls_head.predictions.dense.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "LayerNorm.weight" in key:
                    new_key = "cls_head.predictions.layernorm.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "LayerNorm.bias" in key:
                    new_key = "cls_head.predictions.layernorm.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "seq_relationship" in key:
                    new_key = key.replace("cls", "cls_head")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
            else:
                oneflow_state_dict[key] = self.convert_tensor(oneflow_state_dict.pop(key))
        return oneflow_state_dict

    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.

        Args:
            config_file (str): Path of config file.
        """
        with open(config_file, mode="r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # update default_cfg by config.json
        for k, v in cfg_dict.items():
            if k == "num_hidden_layers":
                self.default_cfg["hidden_layers"] = v
            elif k == "type_vocab_size":
                self.default_cfg["num_tokentypes"] = v
            elif k == "layer_norm_eps":
                self.default_cfg["layernorm_eps"] = v
            elif k in cfg_dict:
                self.default_cfg[k] = v

        # update default_cfg by kwargs
        for k, v in self.kwargs:
            self.default_cfg[k] = v

        # use original BERT residual connection ordering
        self.default_cfg["apply_residual_post_layernorm"] = True


class LoadPretrainedRoberta(LoadPretrainedBert):
    def __init__(self, model, default_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, default_cfg, pretrained_model_path, **kwargs)
        
        """NOTE: base_model_prefix_1 is RoBERTa's prefix in Transformers,
        base_model_prefix_2 is RoBERTa's prefix in LiBai."""
        self.base_model_prefix_1 = "roberta"
        self.base_model_prefix_2 = "roberta"

    def _convert_state_dict(self, torch_state_dict, cfg):
        """Convert torch state dict to flow state dict.

        Args:
            torch_state_dict (OrderedDict): torch state dict.
            cfg (dict): model's default config dict.

        Returns:
            OrderedDict: flow state dict.
        """
        # The converted checkpoint.
        oneflow_state_dict = torch_state_dict.copy()

        # Get configs
        num_heads = cfg.get("num_attention_heads")
        hidden_size = cfg.get("hidden_size")
        layers = cfg.get("hidden_layers")
        head_size = int(hidden_size / num_heads)

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)

        prefix = "roberta." if has_prefix else ""
        index_idx = 3 if has_prefix else 2
        qkv_idx = 6 if has_prefix else 5

        old_keys = oneflow_state_dict.keys()

        for key in list(old_keys):

            # Convert roberta's embedding layers
            if "embeddings" in key:
                if "word_embeddings" in key:
                    new_key = key.replace("word_embeddings", "vocab_embeddings")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "token_type_embeddings" in key:
                    new_key = key.replace("token_type_embeddings", "tokentype_embeddings")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "LayerNorm.weight" in key:
                    new_key = prefix + "encoders.0.input_layernorm.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "LayerNorm.bias" in key:
                    new_key = prefix + "encoders.0.input_layernorm.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                else:
                    oneflow_state_dict[key] = self.convert_tensor(oneflow_state_dict[key])

            # Convert roberta's attention layers
            elif "attention" in key:
                if "self" in key:
                    index = key.split(".")[index_idx]
                    if (
                        prefix + "encoders." + index + ".self_attention.query_key_value.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    q_w = key.replace(key.split(".")[qkv_idx], "query").replace(
                        key.split(".")[qkv_idx + 1], "weight"
                    )
                    k_w = q_w.replace("query", "key")
                    v_w = q_w.replace("query", "value")
                    q_b = q_w.replace("weight", "bias")
                    k_b = k_w.replace("weight", "bias")
                    v_b = v_w.replace("weight", "bias")

                    qkv_w = torch.cat(
                        (
                            oneflow_state_dict.pop(q_w),
                            oneflow_state_dict.pop(k_w),
                            oneflow_state_dict.pop(v_w),
                        ),
                        dim=0,
                    )
                    qkv_b = torch.cat(
                        (
                            oneflow_state_dict.pop(q_b),
                            oneflow_state_dict.pop(k_b),
                            oneflow_state_dict.pop(v_b),
                        ),
                        dim=-1,
                    )

                    qkv_w = self._fix_qkv_ordering(qkv_w, head_size, num_heads)
                    qkv_b = self._fix_qkv_ordering(qkv_b, head_size, num_heads)

                    new_key = (
                        prefix + "encoders." + index + ".self_attention.query_key_value.weight"
                    )
                    oneflow_state_dict[new_key] = self.convert_tensor(qkv_w)

                    new_key = prefix + "encoders." + index + ".self_attention.query_key_value.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(qkv_b)
                elif "output" in key:
                    index = key.split(".")[index_idx]
                    if "dense" in key:
                        if "weight" in key:
                            new_key = prefix + "encoders." + index + ".self_attention.dense.weight"
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                        elif "bias" in key:
                            new_key = prefix + "encoders." + index + ".self_attention.dense.bias"
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                    elif "LayerNorm" in key:
                        if "weight" in key:
                            new_key = (
                                prefix + "encoders." + index + ".post_attention_layernorm.weight"
                            )
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                        elif "bias" in key:
                            new_key = (
                                prefix + "encoders." + index + ".post_attention_layernorm.bias"
                            )
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )

            # Convert roberta's intermediate layers
            elif "intermediate" in key:
                index = key.split(".")[index_idx]
                if (
                    prefix + "encoders." + index + ".mlp.dense_h_to_4h.weight"
                    in oneflow_state_dict.keys()
                ):
                    continue
                if "weight" in key:
                    w = key
                    b = key.replace("weight", "bias")
                    new_key = prefix + "encoders." + index + ".mlp.dense_h_to_4h.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(w))
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(b))

            # Convert roberta's output layers
            elif "output" in key:
                index = key.split(".")[index_idx]
                if "dense.weight" in key:
                    if (
                        prefix + "encoders." + index + ".mlp.dense_4h_to_h.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    new_key = prefix + "encoders." + index + ".mlp.dense_4h_to_h.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(w))
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(b))
                elif "LayerNorm.weight" in key:
                    if (
                        prefix + "encoders." + str(int(index) + 1) + ".input_layernorm.weight"
                        in oneflow_state_dict.keys()
                    ):
                        continue
                    w = key
                    b = w.replace("weight", "bias")
                    if index == str(layers - 1):
                        new_key = prefix + "final_layernorm.weight"
                        oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(w))
                        new_key = new_key.replace("weight", "bias")
                        oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(b))
                        continue
                    new_key = prefix + "encoders." + str(int(index) + 1) + ".input_layernorm.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(w))
                    new_key = new_key.replace("weight", "bias")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(b))

            # Convert roberta's pooler layers
            elif "pooler" in key:
                if "weight" in key:
                    new_key = prefix + "pooler.dense.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "bias" in key:
                    new_key = prefix + "pooler.dense.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))

            # Convert lm_head layers
            elif "lm_head" in key:
                if "layer_norm.weight" in key:
                    new_key = "lm_head.layernorm.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "layer_norm.bias" in key:
                    new_key = "lm_head.layernorm.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "seq_relationship" in key:
                    new_key = key.replace("cls", "cls_head")
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "lm_head.bias" in key:
                    new_key = "lm_head.lm_logits.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                else:
                    oneflow_state_dict[key] = self.convert_tensor(oneflow_state_dict.pop(key))
            else:
                oneflow_state_dict[key] = self.convert_tensor(oneflow_state_dict.pop(key))
        return oneflow_state_dict


class LoadPretrainedGPT2(LoadPretrainedBase):
    def __init__(self, model, default_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, default_cfg, pretrained_model_path, **kwargs)
        
        """NOTE: base_model_prefix_1 is GPT's prefix in Transformers.
        base_model_prefix_2 is GPT's prefix in LiBai."""
        self.base_model_prefix_1 = "transformer"
        self.base_model_prefix_2 = "GPT_model"

    def _convert_state_dict(self, torch_state_dict, cfg):
        """Convert torch state dict to flow state dict.

        Args:
            torch_state_dict (OrderedDict): torch state dict.
            cfg (dict): model's default config dict.

        Returns:
            OrderedDict: flow state dict.
        """
        # The converted checkpoint.
        oneflow_state_dict = torch_state_dict.copy()
        old_keys = list(oneflow_state_dict.keys())

        # Get configs
        num_heads = cfg.get("num_attention_heads")
        hidden_size = cfg.get("hidden_size")
        head_size = int(hidden_size / num_heads)

        # prefix
        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)
        prefix1 = self.base_model_prefix_1 + "." if has_prefix else ""
        prefix2 = "GPT_model." if has_prefix else "GPT_model.transformer."
        layer_idx = 2 if has_prefix else 1

        # Convert Embedding layers.
        new_key = "GPT_model.embeddings.token_embeddings.weight"
        old_keys.remove(prefix1 + "wte.weight")
        oneflow_state_dict[new_key] = self.convert_tensor(
            oneflow_state_dict.pop(prefix1 + "wte.weight")
        )

        new_key = "GPT_model.embeddings.position_embeddings.weight"
        old_keys.remove(prefix1 + "wpe.weight")
        oneflow_state_dict[new_key] = self.convert_tensor(
            oneflow_state_dict.pop(prefix1 + "wpe.weight")
        )

        for key in old_keys:
            keys = key.split(".")
            if layer_idx > len(keys):
                continue
            layer = keys[layer_idx]
            # Convert transformer layers.
            if "h." in key:
                # index = key.split(".")[layer_idx]
                if "ln_1" in key:
                    if "weight" in key:
                        new_key = prefix2 + "layers." + layer + ".input_layernorm.weight"
                    else:
                        new_key = prefix2 + "layers." + layer + ".input_layernorm.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "ln_2" in key:
                    if "weight" in key:
                        new_key = prefix2 + "layers." + layer + ".post_attention_layernorm.weight"
                    else:
                        new_key = prefix2 + "layers." + layer + ".post_attention_layernorm.bias"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
                elif "attn" in key:
                    if "c_attn" in key:
                        if "weight" in key:
                            new_key = (
                                prefix2
                                + "layers."
                                + layer
                                + ".self_attention.query_key_value.weight"
                            )
                        else:
                            new_key = (
                                prefix2 + "layers." + layer + ".self_attention.query_key_value.bias"
                            )
                        qkv = oneflow_state_dict.pop(key)
                        if qkv.ndim > 1:
                            qkv = qkv.transpose(1, 0)
                        qkv = self._fix_qkv_ordering(qkv, head_size, num_heads)
                        oneflow_state_dict[new_key] = self.convert_tensor(qkv)
                    elif "c_proj" in key:
                        if "weight" in key:
                            new_key = prefix2 + "layers." + layer + ".self_attention.dense.weight"
                        elif "bias" in key:
                            new_key = prefix2 + "layers." + layer + ".self_attention.dense.bias"
                        value = oneflow_state_dict.pop(key)
                        if value.ndim > 1:
                            value = value.transpose(1, 0)
                        oneflow_state_dict[new_key] = self.convert_tensor(value)
                elif "mlp" in key:
                    if "c_fc" in key:
                        if "weight" in key:
                            new_key = prefix2 + "layers." + layer + ".mlp.dense_h_to_4h.weight"
                        elif "bias" in key:
                            new_key = prefix2 + "layers." + layer + ".mlp.dense_h_to_4h.bias"
                        value = oneflow_state_dict.pop(key)
                        if value.ndim > 1:
                            value = value.transpose(1, 0)
                        oneflow_state_dict[new_key] = self.convert_tensor(value)
                    elif "c_proj" in key:
                        if "weight" in key:
                            new_key = prefix2 + "layers." + layer + ".mlp.dense_4h_to_h.weight"
                        elif "bias" in key:
                            new_key = prefix2 + "layers." + layer + ".mlp.dense_4h_to_h.bias"
                        value = oneflow_state_dict.pop(key)
                        if value.ndim > 1:
                            value = value.transpose(1, 0)
                        oneflow_state_dict[new_key] = self.convert_tensor(value)
            elif "ln_f" in key:
                if "weight" in key:
                    new_key = prefix2 + "layernorm_f.weight"
                elif "bias" in key:
                    new_key = prefix2 + "layernorm_f.bias"
                oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))
        return oneflow_state_dict

    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.

        Args:
            config_file (str): Path of config file.
        """
        with open(config_file, mode="r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # update default_cfg by config.json
        self.default_cfg["num_layers"] = cfg_dict["n_layer"]
        self.default_cfg["hidden_size"] = cfg_dict["n_embd"]
        self.default_cfg["num_attention_heads"] = cfg_dict["n_head"]
        self.default_cfg["max_seq_length"] = cfg_dict["n_positions"]
        self.default_cfg["embedding_dropout_prob"] = cfg_dict["embd_pdrop"]
        self.default_cfg["attention_dropout_prob"] = cfg_dict["attn_pdrop"]
        self.default_cfg["output_dropout_prob"] = cfg_dict["resid_pdrop"]
        self.default_cfg["layernorm_epsilon"] = cfg_dict["layer_norm_epsilon"]
        self.default_cfg["vocab_size"] = cfg_dict["vocab_size"]
        self.default_cfg["initializer_range"] = cfg_dict["initializer_range"]
        self.default_cfg["ffn_hidden_size"] = cfg_dict.get(
            "n_inner", 4 * self.default_cfg["hidden_size"]
        )

        # update default_cfg by kwargs
        for k, v in self.kwargs:
            self.default_cfg[k] = v


class LoadPretrainedT5(LoadPretrainedBase):
    def __init__(self, model, default_cfg, pretrained_model_path, **kwargs):
        super().__init__(model, default_cfg, pretrained_model_path, **kwargs)
        
        """NOTE: base_model_prefix_1 is T5's prefix in Transformers.
        base_model_prefix_2 is T5's prefix in LiBai."""
        self.base_model_prefix_1 = "transformer"
        self.base_model_prefix_2 = "t5_model"

    def _convert_state_dict(self, torch_state_dict, cfg):
        """Convert torch state dict to flow state dict.
        
        Args:
            torch_state_dict (OrderedDict): torch state dict.
            cfg (dict): model's default config dict.
        
        Returns:
            OrderedDict: flow state dict.
        """
        # The converted checkpoint.
        oneflow_state_dict = torch_state_dict.copy()
        old_keys = list(oneflow_state_dict.keys())
        # Get configs
        num_heads = cfg.get("num_attention_heads")
        hidden_size = cfg.get("hidden_size")
        head_size = cfg.get("head_size", None)
        if head_size == None:
            head_size = int(hidden_size / num_heads)

        has_prefix = any(s.startswith(self.base_model_prefix_1) for s in oneflow_state_dict)
        prefix1 = self.base_model_prefix_1 + "." if has_prefix else ""
        prefix2 = self.base_model_prefix_2 + "." if has_prefix else ""
        encoder_decoder_idx = 1 if has_prefix else 0
        layer_idx1 = 3 if has_prefix else 2
        layer_idx2 = 5 if has_prefix else 4
        op_idx = 6 if has_prefix else 5

        # Convert T5's Embedding layers.
        # NOTE: Transformers' T5 has no position embedding layer.
        new_key = prefix2 + "embedding.word_embeddings.weight"
        old_keys.remove(prefix1 + "shared.weight")
        oneflow_state_dict[new_key] = self.convert_tensor(
            oneflow_state_dict.pop(prefix1 + "shared.weight")
        )

        # Convert T5's final_layer_norm
        new_key = prefix2 + "encoder.final_layernorm.weight"
        old_keys.remove(prefix1 + "encoder.final_layer_norm.weight")
        oneflow_state_dict[new_key] = self.convert_tensor(
            oneflow_state_dict.pop(prefix1 + "encoder.final_layer_norm.weight")
        )
        new_key = prefix2 + "decoder.final_layernorm.weight"
        old_keys.remove(prefix1 + "decoder.final_layer_norm.weight")
        oneflow_state_dict[new_key] = self.convert_tensor(
            oneflow_state_dict.pop(prefix1 + "decoder.final_layer_norm.weight")
        )

        # NOTE: Each layers has no bias in Transformer's T5.
        for key in old_keys:
            keys = key.split(".")
            if layer_idx1 > len(keys) or layer_idx2 > len(keys):
                continue
            layer1 = keys[layer_idx1]
            layer2 = keys[layer_idx2]
            op_name = keys[op_idx]

            if keys[op_idx + 1] == "relative_attention_bias" and keys[op_idx] == "SelfAttention":
                new_key = (
                    prefix2
                    + keys[encoder_decoder_idx]
                    + ".layers.0.self_attention.relative_attention_bias.weight"
                )
                oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(key))

            # Convert T5's Encoder layers.
            if keys[encoder_decoder_idx] == "encoder":
                if op_name == "SelfAttention":
                    new_key = (
                        prefix2
                        + "encoder.layers."
                        + layer1
                        + ".self_attention.query_key_value.weight"
                    )
                    if new_key in oneflow_state_dict.keys():
                        continue
                    q_w = ".".join(keys[: op_idx + 1]) + ".q." + "weight"
                    k_w = ".".join(keys[: op_idx + 1]) + ".k." + "weight"
                    v_w = ".".join(keys[: op_idx + 1]) + ".v." + "weight"
                    qkv_w = torch.cat(
                        (
                            oneflow_state_dict.pop(q_w),
                            oneflow_state_dict.pop(k_w),
                            oneflow_state_dict.pop(v_w),
                        ),
                        dim=0,
                    )
                    qkv_w = self._fix_qkv_ordering(qkv_w, head_size, num_heads, hidden_size)
                    oneflow_state_dict[new_key] = self.convert_tensor(qkv_w)

                    o_w = ".".join(keys[: op_idx + 1]) + ".o." + "weight"
                    new_key = prefix2 + "encoder.layers." + layer1 + ".self_attention.dense.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(o_w))
                elif op_name == "layer_norm":
                    if layer2 == "0":
                        new_key = prefix2 + "encoder.layers." + layer1 + ".input_layernorm.weight"
                        oneflow_state_dict[new_key] = self.convert_tensor(
                            oneflow_state_dict.pop(key)
                        )
                    elif layer2 == "1":
                        new_key = (
                            prefix2
                            + "encoder.layers."
                            + layer1
                            + ".post_attention_layernorm.weight"
                        )
                        oneflow_state_dict[new_key] = self.convert_tensor(
                            oneflow_state_dict.pop(key)
                        )
                elif op_name == "DenseReluDense":
                    if cfg.get("mlp_type") == "t5":
                        if keys[op_idx + 1] == "wi":
                            new_key = (
                                prefix2 + "encoder.layers." + layer1 + ".mlp.dense_h_to_4h.weight"
                            )
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                        elif keys[op_idx + 1] == "wo":
                            new_key = (
                                prefix2 + "encoder.layers." + layer1 + ".mlp.dense_4h_to_h.weight"
                            )
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                    elif cfg.get("mlp_type") == "mt5":
                        if keys[op_idx + 1] == "wi_0":
                            new_key = prefix2 + "encoder.layers." + layer1 + ".mlp.wi_0.weight"
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                        elif keys[op_idx + 1] == "wi_1":
                            new_key = prefix2 + "encoder.layers." + layer1 + ".mlp.wi_1.weight"
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                        elif keys[op_idx + 1] == "wo":
                            new_key = prefix2 + "encoder.layers." + layer1 + ".mlp.wo.weight"
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )

            # Convert T5's decoder Layers.
            elif keys[encoder_decoder_idx] == "decoder":
                if op_name == "SelfAttention":
                    new_key = (
                        prefix2
                        + "decoder.layers."
                        + layer1
                        + ".self_attention.query_key_value.weight"
                    )
                    if new_key in oneflow_state_dict.keys():
                        continue
                    q_w = ".".join(keys[: op_idx + 1]) + ".q." + "weight"
                    k_w = ".".join(keys[: op_idx + 1]) + ".k." + "weight"
                    v_w = ".".join(keys[: op_idx + 1]) + ".v." + "weight"
                    qkv_w = torch.cat(
                        (
                            oneflow_state_dict.pop(q_w),
                            oneflow_state_dict.pop(k_w),
                            oneflow_state_dict.pop(v_w),
                        ),
                        dim=0,
                    )
                    qkv_w = self._fix_qkv_ordering(qkv_w, head_size, num_heads, hidden_size)

                    oneflow_state_dict[new_key] = self.convert_tensor(qkv_w)

                    o_w = ".".join(keys[: op_idx + 1]) + ".o." + "weight"
                    new_key = prefix2 + "decoder.layers." + layer1 + ".self_attention.dense.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(o_w))
                elif op_name == "layer_norm":
                    if layer2 == "0":
                        new_key = prefix2 + "decoder.layers." + layer1 + ".input_layernorm.weight"
                        oneflow_state_dict[new_key] = self.convert_tensor(
                            oneflow_state_dict.pop(key)
                        )
                    elif layer2 == "1":
                        new_key = (
                            prefix2
                            + "decoder.layers."
                            + layer1
                            + ".post_attention_layernorm.weight"
                        )
                        oneflow_state_dict[new_key] = self.convert_tensor(
                            oneflow_state_dict.pop(key)
                        )
                    elif layer2 == "2":
                        new_key = (
                            prefix2
                            + "decoder.layers."
                            + layer1
                            + ".post_cross_attention_layernorm.weight"
                        )
                        oneflow_state_dict[new_key] = self.convert_tensor(
                            oneflow_state_dict.pop(key)
                        )
                elif op_name == "EncDecAttention":
                    new_key = prefix2 + "decoder.layers." + layer1 + ".cross_attention.query.weight"
                    if new_key in oneflow_state_dict.keys():
                        continue
                    q_w = ".".join(keys[: op_idx + 1]) + ".q." + "weight"
                    k_w = ".".join(keys[: op_idx + 1]) + ".k." + "weight"
                    v_w = ".".join(keys[: op_idx + 1]) + ".v." + "weight"

                    q_w = oneflow_state_dict.pop(q_w)
                    kv_w = torch.cat(
                        (
                            oneflow_state_dict.pop(k_w),
                            oneflow_state_dict.pop(v_w),
                        ),
                        dim=0,
                    )
                    q_w = self._fix_qkv_ordering(q_w, head_size, num_heads, hidden_size)
                    kv_w = self._fix_qkv_ordering(kv_w, head_size, num_heads, hidden_size)

                    oneflow_state_dict[new_key] = self.convert_tensor(q_w)
                    new_key = (
                        prefix2 + "decoder.layers." + layer1 + ".cross_attention.key_value.weight"
                    )
                    oneflow_state_dict[new_key] = self.convert_tensor(kv_w)

                    o_w = ".".join(keys[: op_idx + 1]) + ".o." + "weight"
                    new_key = prefix2 + "decoder.layers." + layer1 + ".cross_attention.dense.weight"
                    oneflow_state_dict[new_key] = self.convert_tensor(oneflow_state_dict.pop(o_w))
                elif op_name == "DenseReluDense":
                    if cfg.get("mlp_type") == "t5":
                        if keys[op_idx + 1] == "wi":
                            new_key = (
                                prefix2 + "decoder.layers." + layer1 + ".mlp.dense_h_to_4h.weight"
                            )
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                        elif keys[op_idx + 1] == "wo":
                            new_key = (
                                prefix2 + "decoder.layers." + layer1 + ".mlp.dense_4h_to_h.weight"
                            )
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                    elif cfg.get("mlp_type") == "mt5":
                        if keys[op_idx + 1] == "wi_0":
                            new_key = prefix2 + "decoder.layers." + layer1 + ".mlp.wi_0.weight"
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                        elif keys[op_idx + 1] == "wi_1":
                            new_key = prefix2 + "decoder.layers." + layer1 + ".mlp.wi_1.weight"
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
                        elif keys[op_idx + 1] == "wo":
                            new_key = prefix2 + "decoder.layers." + layer1 + ".mlp.wo.weight"
                            oneflow_state_dict[new_key] = self.convert_tensor(
                                oneflow_state_dict.pop(key)
                            )
        return oneflow_state_dict

    def _load_config_from_json(self, config_file):
        """load config from `config.json`, and update default config.
        
        Args:
            config_file (str): Path of config file.
        """
        with open(config_file, mode="r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        self.default_cfg["vocab_size"] = cfg_dict["vocab_size"]
        self.default_cfg["hidden_size"] = cfg_dict["d_model"]
        self.default_cfg["hidden_layers"] = cfg_dict["num_layers"]
        self.default_cfg["num_attention_heads"] = cfg_dict["num_heads"]
        self.default_cfg["intermediate_size"] = cfg_dict["d_ff"]
        self.default_cfg["hidden_dropout_prob"] = cfg_dict["dropout_rate"]
        self.default_cfg["attention_probs_dropout_prob"] = cfg_dict["dropout_rate"]
        self.default_cfg["max_position_embeddings"] = cfg_dict.get("n_positions", 512)
        self.default_cfg["relative_attention_num_buckets"] = cfg_dict[
            "relative_attention_num_buckets"
        ]
        self.default_cfg["embedding_dropout_prob"] = cfg_dict["dropout_rate"]
        self.default_cfg["initializer_range"] = cfg_dict["initializer_factor"]
        self.default_cfg["layernorm_eps"] = cfg_dict["layer_norm_epsilon"]
        self.default_cfg["head_size"] = cfg_dict["d_kv"]
        
        # update default_cfg by kwargs
        for k, v in self.kwargs:
            self.default_cfg[k] = v
