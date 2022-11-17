# Run distributed inference

This is a tutorial on how to quickly run distributed inference in LiBai from a `huggingface` pretrained model.

## Download model weights

run shell
```shell
mkdir -r data_test/t5_inference_model/
cd data_test/t5_inference_model
wget https://huggingface.co/t5-base/resolve/main/pytorch_model.bin https://huggingface.co/t5-base/resolve/main/config.json https://huggingface.co/t5-base/resolve/main/spiece.model
```

the dir will like this:
```shell
data_test/t5_inference_model
├── config.json
├── pytorch_model.bin
├── spiece.model
```

## run text_generation.py

set `vocab_file` path in `projects/MT5/configs/t5_inference.py`

```python
tokenization.tokenizer = LazyCall(T5Tokenizer)(
    vocab_file="data_test/t5_inference_model/spiece.model",
    add_bos_token=True,
)
```

set your own distributed config in `libai/inference/text_generation.py`

```python
if __name__ == "__main__":
    pipeline = TextGenerationPipeline(
        "projects/MT5/configs/t5_inference.py",
        data_parallel=1,
        tensor_parallel=2,
        pipeline_parallel=2,
        pipeline_stage_id=[0] * 12 + [1] * 12,
        pipeline_num_layers=12 * 2,
        model_path="data_test/t5_inference_model",
        mode="huggingface",
    )

    text = ["summarize: She is a student, She is tall, She loves study"]
    dict1 = pipeline(text)
    if dist.is_main_process():
        print(dict1)
```

To run distributed inference on 2 nodes with total 4 GPUs, 

  in `node0`, run:
  ```bash
  NODE=2 NODE_RANK=1 ADDR=192.168.0.1 PORT=12345 bash tools/infer.sh libai/inference/text_generation.py 2
  ``` 
  `NODE=2` means total number of nodes
  
  `NODE_RANK=0` means current node is node0

  `ADDR=192.168.0.0` means the ip address of node0

  `PORT=12345` means the port of node0

  in `node1`, run:
  ```bash
  NODE=2 NODE_RANK=1 ADDR=192.168.0.1 PORT=12345 bash tools/infer.sh libai/inference/text_generation.py 2
  ``` 
  `NODE=2` means total number of nodes
  
  `NODE_RANK=1` means current node is node1

  `ADDR=192.168.0.0` means the ip address of node0

  `PORT=12345` means the port of node0