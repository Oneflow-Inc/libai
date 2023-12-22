# DALLE2
This project is adapted from [dalle2_pytorch](https://github.com/lucidrains/DALLE2-pytorch); And dalle2_pytorch version=0.15.4 is used following this [colab](https://colab.research.google.com/github/LAION-AI/dalle2-laion/blob/main/notebooks/dalle2_laion_alpha.ipynb).
This project aims at guiding how to transfer pytorch models to oneflow and use distributed inference for new users with [LiBai](https://github.com/Oneflow-Inc/libai), details could be found [here](../../docs/source/notes/How_to_use_model_parallel_in_LiBai.md).

## How to run this project
```sh
cd libai/projects/DALLE2
pip install -r requirements.txt
python3 -m oneflow.distributed.launch \
        --nproc_per_node 4 \
        dalle2_inference.py \
        --save_images    \
        --output_dir  ./outputs  \
        --upsample_scale 4 
```
`--nprec_per_node  4` means this model will be executed on 4 gpus under the model parallel mode.
The output images will be saved to `--output_dir` by setting `--save_images`. The resolution of the generated images are 64x64 by default, and could be resize to 256x256 with `--upsample_scale 4` (and 1024x1024 with `--upsample_scale 16`) by using [SwinIR](https://github.com/JingyunLiang/SwinIR).

At the bottom of the dalle2_inference.py, try feeding different text and see what the model will generated.