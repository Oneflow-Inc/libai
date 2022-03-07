#!/bin/bash -e

# cd to libai project root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 -m unittest discover -f -v -s ./tests
