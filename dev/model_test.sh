#!/bin/bash -e

# cd to libai project root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 -m oneflow.distributed.launch --nproc_per_node 8 -m unittest discover -f -v -s ./tests/models
