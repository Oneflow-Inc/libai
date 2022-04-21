## Use the container (with docker >= 19.03)

```shell
cd docker/
# Build:
docker build --build-arg USER_ID=$UID -t libai:v0 .
# Launch (require GPUs):
docker run --gpus all -it --name libai -v /home/user/:/workspace/ --ipc host --net host libai:v0 /bin/bash
```

You can also use the pre-built LiBai containers (the latest compatible version at time of publication can be pulled with `docker pull l1aoxingyu/libai:v0`).

> NOTE: If your nvidia-driver cannot support `cuda111`, you can build the base image from `cuda102`.

## Install new dependencies

Add the following to `Dockerfile` to make persistent changes.

```shell
RUN sudo apt-get update && sudo apt-get install -y vim
```

Or run them in the container to make temporary changes.
