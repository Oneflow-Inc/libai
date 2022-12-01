## Start training

### Downloading dataset

```shell
mkdir mscoco && cd mscoco
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/libai/Stable_diffusion/00000.tar
mkdir 00000
tar -xvf 00000.tar -C 00000/
```

### install oneflow and libai

oneflow
```shell
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/release/feat-global_ctx/cu112
```

libai installation, refer to [Installation instructions](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html)


### install diffusers

refer to [oneflow diffusers installation](https://github.com/Oneflow-Inc/diffusers/wiki/How-to-Run-OneFlow-Stable-Diffusion)