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

# -----------mock torch, put it in the first line-----------
import oneflow as flow

flow.mock_torch.enable()


import copy # noqa
import sys
sys.path.append("../")
sys.path.append("./")
sys.path.append("./libai/")
import onefx as fx # noqa
from typing import List, Dict, Any # noqa
from oneflow import Tensor, nn  # noqa
from transformers import modeling_utils  # noqa
from transformers.modeling_utils import _load_state_dict_into_model  # noqa
from libai.utils import distributed as dist #noqa



# ---------------- mock _load_state_dict_into_model ------------------
def new_load(model_to_load, state_dict, start_prefix):
    # Convert old format to new format if needed from a PyTorch state_dict
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

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # TODO: add start_prefix judgement
    for k, v in model_to_load.state_dict().items():
        if k in state_dict and v.is_global:
            state_dict[k] = state_dict[k].to_global(
                sbp=flow.sbp.broadcast, placement=flow.env.all_device_placement("cpu")
            )
            state_dict[k] = state_dict[k].to_global(
                sbp=v.sbp,
                placement=flow.placement("cpu", ranks=list(v.placement.ranks)),
            )

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix.
        # We can exit early if there are none in this state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier.
    # Note that `state_dict` is a copy of the argument, so it's safe to delete it.
    del state_dict
    return error_msgs


modeling_utils._load_state_dict_into_model = new_load


# -----------------mock tensor.new_ones() -------------
def flow_ones(self, *args, **kwargs):
    return flow.ones(*args, **kwargs, device=self.device, dtype=self.dtype)


Tensor.new_ones = flow_ones


# -----------------mock tensor.new() ------------------
def flow_zeros(self, *args, **kwargs):
    return flow.zeros(*args, **kwargs, device=self.device, dtype=self.dtype)


Tensor.new = flow_zeros

# ------------------mock nn.functional.softmax---------
temp_func = nn.functional.softmax


def flow_softmax(*args, **kwargs):
    if "dtype" in kwargs:
        _tensor = args[0].to(dtype=kwargs.pop("dtype"))
        return temp_func(_tensor, *args[1:], **kwargs)
    else:
        return temp_func(*args, **kwargs)


nn.functional.softmax = flow_softmax

# =============================================
# -----------------def function----------------
# =============================================

def set_pipeline_stage_id(self, placement):
    for param in self.parameters():
            param.data = param.data.to_global(placement=placement)

nn.Module.set_pipeline_stage_id = set_pipeline_stage_id


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f} Yi{suffix}"

def print_model(model, depth=0, max_depth=2, last_child=False, prefix=''):
    indent = "    "
    stage_str = ""
    if hasattr(model, "layer_idx"):
        layer_idx = getattr(model, "layer_idx")
        stage_idx = getattr(model, "stage_idx")
        same_placement = True
        for path, module in model.named_modules():
            if getattr(module, "layer_idx") != layer_idx:
                same_placement = False
        if same_placement:
            stage_str = f" stage{stage_idx}_ranks{dist.get_layer_placement(layer_idx).ranks} "

    if depth > max_depth:
        return
    if isinstance(model, nn.Module):
        params = sum(p.numel() for p in model.parameters())
        print(indent * depth + ("└─" if last_child else "├─") + prefix + str(model.__class__.__name__) + ": " + stage_str  + sizeof_fmt(params) + " params")
    elif isinstance(model, nn.Sequential):
        print(indent * depth + ("└─" if last_child else "├─") + prefix + str(model.__class__.__name__) + ": " + str(len(list(model.named_children()))) + " modules")
    else:
        print(indent * depth + ("└─" if last_child else "├─") + prefix + str(type(model).__name__))
    for i, (name, child) in enumerate(model.named_children()):
        print_model(child, depth=depth+1, max_depth=max_depth, last_child=i==len(list(model.named_children()))-1, prefix=f'[{name}] ')


def auto_set_pipeline_stage_id(model, pipeline_parallel_size=1):
    # Define a local variable to record the number of repeated and integer layers encountered
    count = 0
    max_depth=1
    name_stage_dict = {}
    # Iterate over all submodules and paths of the model
    for path, module in model.named_modules():
        # Get the name and class of the module
        name = path.split(".")[-1]
        prefix_path = ".".join(path.split(".")[:-1])
        module_cls = type(module)
        
        # Determine if the layer is a number, i.e. if it is possible to be a repeated and integer layer
        if name.isdigit():
            # Determine if the layer has been repeated, i.e. if there is the same path and class in named_modules
            repeated = False
            for n, m in model.named_modules():
                prefix_n = ".".join(n.split(".")[:-1])
                if m is not module and prefix_n == prefix_path and type(m) == module_cls:
                    max_depth = max(len(n.split(".")), max_depth)
                    repeated = True
            if repeated:
                count += 1
                # print(f"Layer {name} with path {path} is repeated. {count}")

        name_stage_dict[path] = max(count-1, 0)
        

    length = (count + pipeline_parallel_size - 1) // pipeline_parallel_size
    param_id_set = set() # skip shared weight param 

    for path, module in model.named_modules():
        # Add to_global to the parameter
        layer_idx = name_stage_dict[path]
        stage_idx = layer_idx // length
        setattr(module, "stage_idx", stage_idx)
        setattr(module, "layer_idx", layer_idx)
        if len(path.split(".")) >= max_depth or len(list(module.named_children())) == 0:
            for param in module.parameters():
                if id(param) not in param_id_set:
                    param.data = param.data.to_global(placement=dist.get_layer_placement(layer_idx))
                else:
                    param_id_set.add(id(param))
    if dist.is_main_process():
        print_model(model, depth=0, max_depth=100 if max_depth==1 else max_depth)
    # Return the modified model
    return model

# ---------------def fx for auto changing placement ----------------------

import inspect
import math
from typing import Tuple, Dict, Optional, Any, Callable, Union
from copy import deepcopy
import traceback
import builtins

_customized_not_wrapped_oneflow_functions = [
    flow.ones_like, 
    flow.zeros_like, 
    flow.randn, 
    flow.randn_like, 
    flow.randint, flow.randint_like,
    flow.device
]

class CustomiziedTracer(fx.Tracer):
    def __init__(self, autowrap_modules = (math, ), autowrap_functions: Tuple[Callable, ...] = (), param_shapes_constant: bool = False, 
                 not_wrapped_oneflow_functions=_customized_not_wrapped_oneflow_functions, input_args=None) -> None:
        super().__init__(autowrap_modules, autowrap_functions, param_shapes_constant, not_wrapped_oneflow_functions)
        self.registered_values = {}
        self.args_iter = iter(input_args)
        
    def to_bool(self, obj: fx.Proxy) -> bool: #override
        if obj.node.name in self.registered_values:
            return self.registered_values[obj.node.name]
        return super().to_bool(obj)
    
    def create_proxy(self, kind: str, target, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                     name: Optional[str] = None, type_expr : Optional[Any] = None,
                     proxy_factory_fn: Callable[[fx.Node], fx.Proxy] = None): #override
        arg_values = []
        for i, arg in enumerate(args):
            # if isinstance(arg, tuple) and isinstance(arg[0], fx.Proxy) and callable(target) 
            #   and list(inspect.signature(target).parameters.keys())[0].startswith('*'):
            #     arg = arg[0]
            if isinstance(arg, tuple):
                has_proxy = [1 if isinstance(item, fx.Proxy) else 0 for item in arg]
                has_proxy = sum(has_proxy)
                current_arg_value = []
                if has_proxy > 0:
                    for proxy in arg:
                        if not isinstance(proxy, fx.Proxy):
                            current_arg_value.append(proxy)
                            continue
                        if not proxy.node.name in self.registered_values:
                            raise ValueError(f"{arg.node.name} cannot be found.")
                        else:
                            current_arg_value.append(self.registered_values[proxy.node.name])
                    arg_values.append(tuple(current_arg_value))
                    continue
            if not isinstance(arg, fx.Proxy):
                arg_values.append(arg)
                continue
            if not arg.node.name in self.registered_values:
                raise ValueError(f"{arg.node.name} cannot be found.")
            else:
                arg_values.append(self.registered_values[arg.node.name])
        
        kwarg_values = {}
        for arg_name, arg in kwargs.items():
            if isinstance(arg, tuple):
                has_proxy = [1 if isinstance(item, fx.Proxy) else 0 for item in arg]
                has_proxy = sum(has_proxy)
                current_arg_value = []
                if has_proxy > 0:
                    for proxy in arg:
                        if not isinstance(proxy, fx.Proxy):
                            current_arg_value.append(proxy)
                            continue
                        if not proxy.node.name in self.registered_values:
                            raise ValueError(f"{arg.node.name} cannot be found.")
                        else:
                            current_arg_value.append(self.registered_values[proxy.node.name])
                    kwarg_values[arg_name] = tuple(current_arg_value)
                    continue
            if not isinstance(arg, fx.Proxy):
                kwarg_values[arg_name] = arg
                continue
            if not arg.node.name in self.registered_values:
                raise ValueError(f"{arg.node.name} cannot be found.")
            else:
                kwarg_values[arg_name] = self.registered_values[arg.node.name]
        
        assert kind != "call_function" or callable(target)
        
        with fx.fx_no_wrap_context(self):
            if kind == "call_function":
                result_value = target(*arg_values, **kwarg_values)
            elif kind == "call_method":
                self_obj, *args_tail = arg_values

                # Execute the method and return the result
                assert isinstance(target, str)
                method = getattr(self_obj, target)
                result_value = method(*args_tail, **kwarg_values)
            elif kind == "call_module":
                assert isinstance(target, str)
                submod = self.fetch_attr(target)
                
                result_value = submod(*arg_values, **kwarg_values)
            elif kind == "placeholder":
                assert isinstance(target, str)
                if target.startswith('*'):
                    # For a starred parameter e.g. `*args`, retrieve all
                    # remaining values from the args list.
                    result_value = list(self.args_iter)
                else:
                    try:
                        result_value = next(self.args_iter)
                    except StopIteration as si:
                        raise RuntimeError(f'Expected positional argument for parameter {target}, but one was not passed in!') from si
            elif kind == "get_attr":
                assert isinstance(target, str)
                result_value = self.fetch_attr(target)
            elif kind == "output":
                result_value = arg_values[0]
            elif kind == "root":
                raise NotImplementedError
            else:
                raise NotImplementedError
        
        if isinstance(result_value, fx.Proxy):
            if result_value.node.name in self.registered_values:
                result_value = self.registered_values[result_value.node.name]
            else:
                raise ValueError("Got a proxy object when running with original values.")
        
        if not self.fx_no_wrap:
            result_proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
            self.registered_values[result_proxy.node.name] = result_value
        return result_proxy
    
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        if self.fx_no_wrap:
            return attr_val
        def maybe_get_proxy_for_attr(
            attr_val, collection_to_search, parameter_proxy_cache
        ):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if (
                            "proxy_factory_fn"
                            in inspect.signature(self.create_proxy).parameters
                        ):
                            kwargs["proxy_factory_fn"] = (
                                None
                                if not self.param_shapes_constant
                                else lambda node: fx.ParameterProxy(
                                    self, node, n, attr_val
                                )
                            )
                        val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None

        if isinstance(attr_val, flow.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_parameters(), parameter_proxy_cache
            )
            if maybe_parameter_proxy is not None:
                if not maybe_parameter_proxy.node.name in self.registered_values:
                    self.registered_values[maybe_parameter_proxy.node.name] = attr_val
                return maybe_parameter_proxy

        if self.proxy_buffer_attributes and isinstance(attr_val, flow.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_buffers(), parameter_proxy_cache
            )
            if maybe_buffer_proxy is not None:
                if not maybe_buffer_proxy.node.name in self.registered_values:
                    self.registered_values[maybe_buffer_proxy.node.name] = attr_val
                return maybe_buffer_proxy

        return attr_val
    
    def call_module(
        self,
        m: flow.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any: # override
        if self.fx_no_wrap:
            return forward(*args, **kwargs)
        else:
            return super().call_module(m, forward, args, kwargs)
        
    def trace(
        self,
        root: Union[flow.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> fx.Graph: # override
        self.module = root
        return super().trace(root, concrete_args)
    
    def fetch_attr(self, target : str):
        target_atoms = target.split('.')
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        if not isinstance(attr_itr, fx.Proxy):
            return attr_itr
        if attr_itr.node.name in self.registered_values:
            return self.registered_values[attr_itr.node.name]
        
        raise ValueError(f"No attr <{target}> was found.")
        

def customized_symbolic_trace(
    root: Union[flow.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
    input_args=None
) -> fx.GraphModule:
    tracer = CustomiziedTracer(input_args=input_args)
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, flow.nn.Module) else root.__name__
    )
    return fx.GraphModule(tracer.root, graph, name)

class AutoPlacementInterpreter(fx.Interpreter):
    def __init__(self, mod : flow.nn.Module, concrete_args=None, input_args=None):
        gm = customized_symbolic_trace(mod, concrete_args=concrete_args, input_args=input_args)
        super().__init__(gm)

        self.global_infos : Dict[int, Dict[int, Any]] = {}
        self.node_id = 0

    def run(self, *args) -> Any:
        return_val = super().run(*args)
        return return_val

    def run_node(self, n : fx.Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        global_info_to_replace = None
        max_rank_sum = -1
        for arg in args:
            if not isinstance(arg, flow.Tensor):
                continue
            if arg.is_local or len(arg.placement.ranks) == 0:
                continue
            placement = arg.placement
            sbp = arg.sbp
            # print(sum(placement.ranks))
            if max_rank_sum < sum(placement.ranks):
                max_rank_sum = sum(placement.ranks)
                global_info_to_replace = (placement, sbp)
            # elif max_rank_sum == sum(placement.ranks) and zip(placement_to_replace.ranks, placement.ranks).all(lambda x, y: x == y):
            #     raise ValueError("There is two different placements with same rank sum. "
            #                             + f"They are {placement_to_replace} and {placement}.")
        
        if max_rank_sum == -1:
            self.node_id += 1
            return_val = super().run_node(n)
            return return_val
        
        for arg_id in range(len(args)):
            if isinstance(arg, flow.Tensor) and sum(arg.placement.ranks) < max_rank_sum:
                self.global_infos.setdefault(self.node_id, {})[arg_id] = global_info_to_replace
                n.update_arg(arg_id, args[arg_id].to_global(global_info_to_replace[0], global_info_to_replace[1]))

        return_val = super().run_node(n)
        return return_val


def add_auto_placement(model: flow.nn.Module, global_info_dict: Dict[int, Dict[int, List[int]]], concrete_args=None, input_args=None) -> flow.nn.Module:
    model = copy.deepcopy(model)
    fx_model: fx.GraphModule = customized_symbolic_trace(model, concrete_args=concrete_args, input_args=input_args)

    for node_id, node in enumerate(fx_model.graph.nodes):
        if not node_id in global_info_dict:
            continue
        
        for idx, arg in enumerate(node.args):
            if not idx in global_info_dict[node_id]:
                continue
            global_info = global_info_dict[node_id][idx]
            new_node = fx.Node(fx_model.graph, f"auto_placement_{node_id}_{idx}", "call_function", flow.to_global, (arg, global_info[0], global_info[1]), {})
            node.prepend(new_node)
            node.update_arg(idx, new_node)

    fx_model.graph.lint()
    fx_model.recompile()
    return fx.Interpreter(fx_model)

fx.wrap(len)
def compile_auto_placement(model: flow.nn.Module, concrete_args=None, **kwargs):
    with fx.global_wrap([dist.get_nd_sbp, dist.same_sbp], dist):
        with fx.global_wrap([flow.finfo], flow):
            if concrete_args is None:
                all_args = inspect.signature(model.forward).parameters
                concrete_args = {}
                for arg_name, param in all_args.items():
                    if not arg_name in kwargs and param.default != inspect._empty:
                        concrete_args.update({arg_name:param.default})
            
            interpret = AutoPlacementInterpreter(model, concrete_args=concrete_args, input_args=list(kwargs.values()) + list(concrete_args.values()))
            interpret.run(*(kwargs.values()))
            model = add_auto_placement(model, interpret.global_infos, concrete_args, input_args=list(kwargs.values()) + list(concrete_args.values()))
            return model

# b = flow.ones(
#     (2,2), 
#     sbp=[flow.sbp.broadcast, flow.sbp.broadcast], 
#     placement=flow.placement("cuda", ranks=[[2], [3]])
# )
# demo_module = demoModule()
# interpret = AutoPlacementInterpreter(demo_module)
# c = interpret.run(b)
# model = add_auto_placement(demo_module, interpret.global_infos)
# print(model.code)
# print(model(b))