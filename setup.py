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

import os
import subprocess
import sys

import pybind11
from setuptools import Extension, find_packages, setup

version = "0.0.1.3"
package_name = "LiBai"
cwd = os.path.dirname(os.path.abspath(__file__))

sha = "Unknown"
try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
except Exception:
    pass


def write_version_file():
    version_path = os.path.join(cwd, "libai", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")


if sys.version_info < (3,):
    sys.exit("Sorry, Python3 is required for LiBai.")

requirements = [
    "boto3",
    "botocore",
    "cloudpickle",
    "flowvision>=0.0.6",
    "hydra-core",
    "nltk",
    "numpy",
    "omegaconf",
    "oneflow>=0.6.0",
    "Pygments",
    "PyYAML",
    "regex",
    "requests",
    "sentencepiece>=0.1",
    "tabulate",
    "termcolor",
    "tqdm",
    "pybind11",
    "portalocker",
    "flake8==3.8.1 ",
    "isort==5.10.1",
    "black==21.4b2",
]

extensions = [
    Extension(
        "libai.data.data_utils.helpers",
        sources=["libai/data/data_utils/helpers.cpp"],
        extra_compile_args=[
            "-O3",
            "-Wall",
            "-shared",
            "-std=c++11",
            "-fPIC",
            "-fdiagnostics-color",
        ],
        include_dirs=[pybind11.get_include()],
    ),
]

if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")

    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()

    with open("LICENSE", "r", encoding="utf-8") as f:
        license = f.read()

    write_version_file()

    setup(
        name=package_name,
        version=version,
        description="Toolkit for Pretraining Models with OneFlow",
        long_description=readme,
        license=license,
        install_requires=requirements,
        packages=find_packages(),
        ext_modules=extensions,
        test_suite="tests",
    )
