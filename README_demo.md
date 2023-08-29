
## build docker image
```
cd docker
docker build -f demo-dockerfile -t libai-demo:v0.1 .
```
## launch docker container
```
docker run -it --rm --runtime=nvidia --privileged \
  --network host --gpus=all \
  --ipc=host \
  libai-demo:v0.1 \
  bash
```

## pretrain
in folder `/workspace/libai_repo`
- 1 node 1 device: `./1x1x1.sh`
- 1 node 8 devices: `./2x2x2.sh`, data parallel x 2, tensor parallel x 2, pipeline parallel x 2
- 2 node 16 devices
  - node 0: `./node0_4x2x2.sh`
  - node 1: `./node1_4x2x2.sh`
