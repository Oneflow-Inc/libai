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

<<<<<<< HEAD
import pydoc
=======
>>>>>>> main
import ast
import builtins
import importlib
import inspect
import logging
import os
<<<<<<< HEAD
=======
import pydoc
>>>>>>> main
import uuid
from collections import abc
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import is_dataclass
<<<<<<< HEAD
from typing import List, Tuple, Union, Any
=======
from typing import Any, List, Tuple, Union

>>>>>>> main
import cloudpickle
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["LazyCall", "LazyConfig"]


def locate(name: str) -> Any:
    """
    Locate and return an object ``x`` using an input string ``{x.__module__}.{x.__qualname__}``,
    such as "module.submodule.class_name".
    Raise Exception if it cannot be found.
    """
    obj = pydoc.locate(name)

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.
    if obj is None:
        try:
            # from hydra.utils import get_method - will print many errors
            from hydra.utils import _locate
        except ImportError as e:
            raise ImportError(f"Cannot dynamically locate object {name}!") from e
        else:
            obj = _locate(name)  # it raises if fails

    return obj


def _convert_target_to_string(t: Any) -> str:
    """
    Inverse of ``locate()``.
    Args:
        t: any object with ``__module__`` and ``__qualname__``
    """
    module, qualname = t.__module__, t.__qualname__

    # Compress the path to this object, e.g. ``module.submodule._impl.class``
    # may become ``module.submodule.class``, if the later also resolves to the same
    # object. This simplifies the string, and also is less affected by moving the
    # class implementation.
    module_parts = module.split(".")
    for k in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:k])
        candidate = f"{prefix}.{qualname}"
        try:
            if locate(candidate) is t:
                return candidate
        except ImportError:
            pass
    return f"{module}.{qualname}"


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.
    LazyCall object has to be called with only keyword arguments. Positional
    arguments are not yet supported.
    Examples:
    ::
        from libai.config import instantiate, LazyCall
        layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
        layer_cfg.out_channels = 64   # can edit it afterwards
        layer = instantiate(layer_cfg)
    """

    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(
                f"target of LazyCall must be a callable or defines a callable! Got {target}"
            )
        self._target = target

    def __call__(self, **kwargs):
        if is_dataclass(self._target):
            # omegaconf object cannot hold dataclass type
            # https://github.com/omry/omegaconf/issues/784
            target = _convert_target_to_string(self._target)
        else:
            target = self._target
        kwargs["_target_"] = target

        return DictConfig(content=kwargs, flags={"allow_objects": True})


def _visit_dict_config(cfg, func):
    """
    Apply func recursively to all DictConfig in cfg.
    """
    if isinstance(cfg, DictConfig):
        func(cfg)
        for v in cfg.values():
            _visit_dict_config(v, func)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _visit_dict_config(v, func)


def _validate_py_syntax(filename):
    # see also https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
    with open(filename, "r", encoding="utf-8") as f:
        # Setting encoding explicitly to resolve coding issue on windows
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError(f"Config file {filename} has syntax error!") from e


def _cast_to_config(obj):
    # if given a dict, return DictConfig instead
    if isinstance(obj, dict):
        return DictConfig(obj, flags={"allow_objects": True})
    return obj


_CFG_PACKAGE_NAME = "libai._cfg_loader"
"""
A namespace to put all imported config into.
"""


def _random_package_name(filename):
    # generate a random package name when loading config files
    return _CFG_PACKAGE_NAME + str(uuid.uuid4())[:4] + "." + os.path.basename(filename)


@contextmanager
def _patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
       e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. support other storage system through PathManager
    4. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):
        cur_file = os.path.dirname(original_file)
        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)
        cur_name = relative_import_path.lstrip(".")
        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)
        # NOTE: directory import is not handled. Because then it's unclear
        # if such import should produce python module or DictConfig. This can
        # be discussed further if needed.
        if not cur_file.endswith(".py"):
            cur_file += ".py"
        if not os.path.isfile(cur_file):
            raise ImportError(
                f"Cannot import name {relative_import_path} from "
                f"{original_file}: {cur_file} has to exist."
            )
        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            level != 0
            and globals is not None
            and (globals.get("__package__", "") or "").startswith(_CFG_PACKAGE_NAME)
        ):
            cur_file = find_relative_file(globals["__file__"], name, level)
            _validate_py_syntax(cur_file)
            spec = importlib.machinery.ModuleSpec(
                _random_package_name(cur_file), None, origin=cur_file
            )
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file
            with open(cur_file, "r", encoding="utf-8") as f:
                content = f.read()
            exec(compile(content, cur_file, "exec"), module.__dict__)
            for name in fromlist:  # turn imported dict into DictConfig automatically
                val = _cast_to_config(module.__dict__[name])
                module.__dict__[name] = val
            return module
        return old_import(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import


class LazyConfig:
    """
    Provide methods to save, load, and overrides an omegaconf config object
    which may contain definition of lazily-constructed objects.
    """

    @staticmethod
    def load_rel(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Similar to :meth:`load()`, but load path relative to the caller's
        source file.
        This has the same functionality as a relative import, except that this method
        accepts filename as a string, so more characters are allowed in the filename.
        """
        caller_frame = inspect.stack()[1]
        caller_fname = caller_frame[0].f_code.co_filename
        assert caller_fname != "<string>", "load_rel Unable to find caller"
        caller_dir = os.path.dirname(caller_fname)
        filename = os.path.join(caller_dir, filename)
        return LazyConfig.load(filename, keys)

    @staticmethod
    def load(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Load a config file.
        Args:
            filename: absolute path or relative path w.r.t. the current working directory
            keys: keys to load and return. If not given, return all keys
                (whose values are config objects) in a dict.
        """
        has_keys = keys is not None
        filename = filename.replace("/./", "/")  # redundant
        if os.path.splitext(filename)[1] not in [".py", ".yaml", ".yml"]:
            raise ValueError(f"Config file {filename} has to be a python or yaml file.")
        if filename.endswith(".py"):
            _validate_py_syntax(filename)

            with _patch_import():
                # Record the filename
                module_namespace = {
                    "__file__": filename,
                    "__package__": _random_package_name(filename),
                }
                with open(filename, "r", encoding="utf-8") as f:
                    content = f.read()
                # Compile first with filename to:
                # 1. make filename appears in stacktrace
                # 2. make load_rel able to find its parent's (possibly remote) location
                exec(compile(content, filename, "exec"), module_namespace)

            ret = module_namespace
        else:
            with open(filename, "r", encoding="utf-8") as f:
                obj = yaml.unsafe_load(f)
            ret = OmegaConf.create(obj, flags={"allow_objects": True})

        if has_keys:
            if isinstance(keys, str):
                return _cast_to_config(ret[keys])
            else:
                return tuple(_cast_to_config(ret[a]) for a in keys)
        else:
            if filename.endswith(".py"):
                # when not specified, only load those that are config objects
                ret = DictConfig(
                    {
                        name: _cast_to_config(value)
                        for name, value in ret.items()
                        if isinstance(value, (DictConfig, ListConfig, dict))
                        and not name.startswith("_")
                    },
                    flags={"allow_objects": True},
                )
            return ret

    @staticmethod
    def save(cfg, filename: str):
        """
        Save a config object to a yaml file.
        Note that when the config dictionary contains complex objects (e.g. lambda),
        it can't be saved to yaml. In that case we will print an error and
        attempt to save to a pkl file instead.
        Args:
            cfg: an omegaconf config object
            filename: yaml file name to save the config file
        """
        logger = logging.getLogger(__name__)
        try:
            cfg = deepcopy(cfg)
        except Exception:
            pass
        else:
            # if it's deep-copyable, then...
            def _replace_type_by_name(x):
                if "_target_" in x and callable(x._target_):
                    try:
                        x._target_ = _convert_target_to_string(x._target_)
                    except AttributeError:
                        pass

            # not necessary, but makes yaml looks nicer
            _visit_dict_config(cfg, _replace_type_by_name)

        save_pkl = False
        try:
            dict = OmegaConf.to_container(cfg, resolve=False)
<<<<<<< HEAD
            dumped = yaml.dump(
                dict, default_flow_style=None, allow_unicode=True, width=9999
            )
=======
            dumped = yaml.dump(dict, default_flow_style=None, allow_unicode=True, width=9999)
>>>>>>> main
            with open(filename, "w") as f:
                f.write(dumped)

            try:
                _ = yaml.unsafe_load(dumped)  # test that it is loadable
            except Exception:
                logger.warning(
                    "The config contains objects that cannot serialize to a valid yaml. "
                    f"{filename} is human-readable but cannot be loaded."
                )
                save_pkl = True
        except Exception:
            logger.exception("Unable to serialize the config to yaml. Error:")
            save_pkl = True

        if save_pkl:
            new_filename = filename + ".pkl"
            try:
                # retry by pickle
                with open(new_filename, "wb") as f:
                    cloudpickle.dump(cfg, f)
                logger.warning(f"Config is saved using cloudpickle at {new_filename}.")
            except Exception:
                pass

    @staticmethod
    def apply_overrides(cfg, overrides: List[str]):
        """
        In-place override contents of cfg.
        Args:
            cfg: an omegaconf config object
            overrides: list of strings in the format of "a=b" to override configs.
                See https://hydra.cc/docs/next/advanced/override_grammar/basic/
                for syntax.
        Returns:
            the cfg object
        """

        def safe_update(cfg, key, value):
            parts = key.split(".")
            for idx in range(1, len(parts)):
                prefix = ".".join(parts[:idx])
                v = OmegaConf.select(cfg, prefix, default=None)
                if v is None:
                    break
                if not OmegaConf.is_config(v):
                    raise KeyError(
                        f"Trying to update key {key}, but {prefix} "
                        f"is not a config, but has type {type(v)}."
                    )
            OmegaConf.update(cfg, key, value, merge=True)

        from hydra.core.override_parser.overrides_parser import OverridesParser

        parser = OverridesParser.create()
        overrides = parser.parse_overrides(overrides)
        for o in overrides:
            key = o.key_or_group
            value = o.value()
            if o.is_delete():
                # TODO support this
                raise NotImplementedError("deletion is not yet a supported override")
            safe_update(cfg, key, value)
        return cfg

    @staticmethod
    def to_py(cfg, prefix: str = "cfg."):
        """
        Try to convert a config object into Python-like pseudo code.
        Note that perfect conversion is not always possible. So the returned
        results are mainly meant to be human-readable, and not meant to be executed.
        Args:
            cfg: an omegaconf config object
            prefix: root name for the resulting code (default: "cfg.")
        Returns:
            str of formatted Python code
        """
        import black

        cfg = OmegaConf.to_container(cfg, resolve=True)

        def _to_str(obj, prefix=None, inside_call=False):
            if prefix is None:
                prefix = []
            if isinstance(obj, abc.Mapping) and "_target_" in obj:
                # Dict representing a function call
                target = _convert_target_to_string(obj.pop("_target_"))
                args = []
                for k, v in sorted(obj.items()):
                    args.append(f"{k}={_to_str(v, inside_call=True)}")
                args = ", ".join(args)
                call = f"{target}({args})"
                return "".join(prefix) + call
            elif isinstance(obj, abc.Mapping) and not inside_call:
                # Dict that is not inside a call is a list of top-level config objects that we
                # render as one object per line with dot separated prefixes
                key_list = []
                for k, v in sorted(obj.items()):
                    if isinstance(v, abc.Mapping) and "_target_" not in v:
                        key_list.append(_to_str(v, prefix=prefix + [k + "."]))
                    else:
                        key = "".join(prefix) + k
                        key_list.append(f"{key}={_to_str(v)}")
                return "\n".join(key_list)
            elif isinstance(obj, abc.Mapping):
                # Dict that is inside a call is rendered as a regular dict
                return (
                    "{"
                    + ",".join(
                        f"{repr(k)}: {_to_str(v, inside_call=inside_call)}"
                        for k, v in sorted(obj.items())
                    )
                    + "}"
                )
            elif isinstance(obj, list):
<<<<<<< HEAD
                return (
                    "["
                    + ",".join(_to_str(x, inside_call=inside_call) for x in obj)
                    + "]"
                )
=======
                return "[" + ",".join(_to_str(x, inside_call=inside_call) for x in obj) + "]"
>>>>>>> main
            else:
                return repr(obj)

        py_str = _to_str(cfg, prefix=[prefix])
        try:
            return black.format_str(py_str, mode=black.Mode())
        except black.InvalidInput:
            return py_str
