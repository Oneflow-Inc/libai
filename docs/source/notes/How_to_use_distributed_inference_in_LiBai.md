# Detailed instruction for using distributed inference in LiBai

If you want to using distributed inference in LiBai from pretrained `pytorch` model, you can refer to [DALLE2 inference doc](https://github.com/Oneflow-Inc/libai/blob/main/docs/source/notes/How_to_use_model_parallel_in_LiBai.md). And [Chinese doc for distributed inference](https://github.com/Oneflow-Inc/libai/discussions/386) is also available.

Here we introduce how to use distributed inference in LiBai:

## Check `model.py`

check your `model.py` first:
1. Ensure There are `libai.layers` in your `model.py`:
   ```python
    # NOTE: you don't need to import all layers from libai, if you only use libai.layers.Linear 
    # in your `model.py`, you model will run model/pipeline parallel only in `Linear` layers 
    from libai.layers import (
        Linear, 
        LayerNorm,
        ...
    )
   ```
2. If you want to run pipeline parallel in LiBai, you should additionally insert code `x = x.to_global(placement=target_tensor.placement)` in your `model.forward()`. 
It is equal to torch code `x.to(cuda_device)`, which move tensor from gpuA to gpuB. There are many examples in LiBai: [example1](https://github.com/Oneflow-Inc/libai/blob/92dbe7c1b1496290e32e595f8473f9288ea1886e/projects/MT5/layers/attention_layer.py#L220), [example2](https://github.com/Oneflow-Inc/libai/blob/92dbe7c1b1496290e32e595f8473f9288ea1886e/projects/MT5/layers/attention_layer.py#L156) ...
   
    If you don't know where to insert code, you can run your code first, and the it will raise bug in the line which needed `to_global`. 
    for example:

    ```shell
      File "libai/libai/layers/layer_norm.py", line 129, in forward   

        return flow._C.rms_layer_norm(hidden_states, self.weight, self.l2norm_epsilon)    RuntimeError: return flow._C.rms_layer_norm(hidden_states, self.weight, self.l2norm_epsilon)RuntimeErrorExpected all tensors to be on the same placement, but found at least two placements, oneflow.placement(type="cuda", ranks=[0, 1]) (positional 0) and oneflow.placement(type="cuda", ranks=[2, 3]) (positional 1)!
    ```

## Build `config.py`

If your model is Trained from LiBai, you can use the same `config.py` from training. refer to [Couplets](https://github.com/Oneflow-Inc/libai/tree/main/projects/Couplets#inference) for more details

If your model is Trainer from other framework, you should build your own `inference_config.py`, you can refer to [`dalle2_config.py`](https://github.com/Oneflow-Inc/libai/blob/main/projects/DALLE2/configs/dalle2_config.py) and [`t5_inference_config.py `](https://github.com/Oneflow-Inc/libai/blob/main/projects/MT5/configs/t5_inference.py)

## Refine `pipeline_inference.py`

The base class [libai/inference/basic.py](https://github.com/Oneflow-Inc/libai/blob/main/libai/inference/basic.py) is already provided in `LiBai` , 
Users only need to overload the functions they need. refer to [text_generation.py](https://github.com/Oneflow-Inc/libai/blob/main/libai/inference/text_generation.py)

If your model is trained from `LiBai`, it will be easy to use, you can refer to [distribute_infer.py](https://github.com/Oneflow-Inc/libai/blob/main/projects/Couplets/distribute_infer.py) in [Couplets](https://github.com/Oneflow-Inc/libai/tree/main/projects/Couplets)

If your model is trained from other framework, you need to build your own `model_loader` to load your model weights in LiBai, refer to [model_loader](https://libai.readthedocs.io/en/latest/notes/How_to_load_huggingface%27s_pretrained_model_in_libai.html) for more details

Give a simple example,  the function overloaded in `LiBai`:
```python
from libai.inference.basic import BasePipeline
from libai.utils import distributed as dist

class MyPipeline(BasePipeline):
    def _parse_parameters(self, **pipeline_parameters):
        # By overloading this function, the input parameters in MyPipeline.__call__() hand out to preprocess/forward/postprocess stages of inference.
        preprocess_params = {
            "preprocess_param1": pipeline_parameters["preprocess_param1"],
            "preprocess_param2": pipeline_parameters["preprocess_param2"],
        }
        forward_params = {
            "forward_param": pipeline_parameters["forward_param"]
        }
        postprocess_params = {
            "postprocess_param": pipeline_parameters["postprocess_param"]
        }
        return preprocess_params, forward_params, postprocess_params

    def load_pretrain_weight(self, libai_cfg_model, model_path, mode="myloader"):
        # load your pretrain weight in this functor
        # set your own "MyLoader" if your model is pretrained from other framework
        # set mode to "libai" if your model is pretrained from libai
        if mode == "myloader":
            import MyLoader

            model_loader = MyLoader(
                libai_cfg_model,
                libai_cfg_model.cfg,
                model_path,
                ...,
            )
            return model_loader.load()
        else:
            return super().load_pretrain_weight(
                libai_cfg_model,
                model_path,
                mode=mode,
            )

    def preprocess(self, inputs, preprocess_param1, preprocess_param2, **kwargs):
        ...
        # model_input_dict: {"key1": flow.Tensor1, ...}
        return model_input_dict
    
    def forward(self, model_input_dict, forward_param, **kwargs):
        ...
        model_output_dict = self.model(**model_input_dict)
        return model_output_dict

    def postprocess(self, model_output_dict, postprocess_param, **kwargs):
        ...
        return out_dict

if __name__ == "__main__":
    pipeline = MyPipeline(
        "path/to/myconfig.py",
        data_parallel=1,
        tensor_parallel=...,
        pipeline_parallel=...,
        pipeline_stage_id=...,
        pipeline_num_layers=...,
        model_path=...,
        mode=...,
    )
    out = pipeline(
        input_text=..., 
        preprocess_param1=..., 
        preprocess_param2=...,
        forward_param=...,
        postprocess_param=...,
    )
    if dist.is_main_process():
        print(out)
```

## Distributed run `pipeline_inference.py`

To run model on 2 nodes with total 4 GPUs, 

  in `node0`, run:
  ```bash
  NODE=2 NODE_RANK=1 ADDR=192.168.0.1 PORT=12345 bash tools/infer.sh pipeline_inference.py 2
  ``` 
  `NODE=2` means total number of nodes
  
  `NODE_RANK=0` means current node is node0

  `ADDR=192.168.0.0` means the ip address of node0

  `PORT=12345` means the port of node0

  in `node1`, run:
  ```bash
  NODE=2 NODE_RANK=1 ADDR=192.168.0.1 PORT=12345 bash tools/infer.sh pipeline_inference.py 2
  ``` 
  `NODE=2` means total number of nodes
  
  `NODE_RANK=1` means current node is node1

  `ADDR=192.168.0.0` means the ip address of node0

  `PORT=12345` means the port of node0
