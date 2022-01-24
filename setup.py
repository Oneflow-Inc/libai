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

import sys

import pybind11
from setuptools import Extension, find_packages, setup

if sys.version_info < (3,):
    sys.exit("Sorry, Python3 is required for LiBai.")

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

with open("requirements.txt") as f:
    reqs = f.read()


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

setup(
    name="LiBai",
    version="0.0.1",
    description="Toolkit for Pretraining Models with OneFlow",
    long_description=readme,
    license=license,
    install_requires=reqs.strip().split("\n"),
    packages=find_packages(),
    ext_modules=extensions,
    test_suite="tests",
)
