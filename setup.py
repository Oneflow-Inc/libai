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

import glob
import os
import shutil
import subprocess
import sys
from os import path
from typing import List

from setuptools import Extension, find_packages, setup

version = "0.2.0"
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


def get_pybind11():
    import pybind11 as pb

    return pb


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
        include_dirs=[get_pybind11().get_include()],
    ),
]


def get_libai_configs() -> List[str]:
    """
    Return a list of configs to include in package for model zoo.
    """
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
    destination = path.join(path.dirname(path.realpath(__file__)), "libai", "config", "configs")
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if path.exists(source_configs_dir):
        if path.islink(destination):
            os.unlink(destination)
        elif path.isdir(destination):
            shutil.rmtree(destination)

    if not path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)
    config_paths = glob.glob("configs/**/*.py", recursive=True)
    return config_paths


if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")

    with open("LICENSE", "r", encoding="utf-8") as f:
        license = f.read()

    write_version_file()

    setup(
        name=package_name,
        version=version,
        description="Toolkit for Pretraining Models with OneFlow",
        license=license,
        install_requires=[
            "boto3",
            "botocore",
            "cloudpickle",
            "flowvision==0.1.0",
            "wget",
            "hydra-core",
            "nltk",
            "numpy",
            "omegaconf==2.1.0",
            "Pygments",
            "PyYAML",
            "jieba",
            "regex",
            "requests",
            "scipy",
            "sentencepiece>=0.1",
            "tabulate",
            "termcolor",
            "tqdm",
            "pybind11",
            "portalocker",
            "flake8==3.8.1 ",
            "isort==5.10.1",
            "black==21.4b ",
            "autoflake",
            "tensorboardX",
            "pytest",
        ],
        packages=find_packages(),
        package_data={"libai.config": get_libai_configs()},
        ext_modules=extensions,
        test_suite="tests",
    )
