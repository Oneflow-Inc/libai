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
from pathlib import Path

import numpy as np
import oneflow as flow
from flowvision import transforms
from oneflow.utils.data import Dataset
from PIL import Image

from libai.data.structures import DistTensorData, Instance


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        tokenizer_pretrained_folder=None,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        if tokenizer_pretrained_folder:
            self.tokenizer = self.tokenizer.from_pretrained(
                tokenizer_pretrained_folder[0], subfolder=tokenizer_pretrained_folder[1]
            )

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if self.class_data_root and np.random.rand() > 0.5:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            instance_images = self.image_transforms(class_image)
            input_ids = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="np",
            ).input_ids
        else:
            instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            instance_images = self.image_transforms(instance_image)
            input_ids = self.tokenizer(
                self.instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="np",
            ).input_ids

        return Instance(
            pixel_values=DistTensorData(instance_images.to(dtype=flow.float32)),
            input_ids=DistTensorData(flow.tensor(input_ids[0])),
        )


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class TXTDataset(Dataset):
    def __init__(
        self,
        foloder_name,
        tokenizer,
        tokenizer_pretrained_folder=None,
        thres=0.2,
        size=512,
        center_crop=False,
    ):
        print(f"Loading folder data from {foloder_name}.")
        self.image_paths = []
        self.tokenizer = tokenizer
        if tokenizer_pretrained_folder:
            self.tokenizer = self.tokenizer.from_pretrained(
                tokenizer_pretrained_folder[0], subfolder=tokenizer_pretrained_folder[1]
            )
        for each_file in os.listdir(foloder_name):
            if each_file.endswith(".jpg"):
                self.image_paths.append(os.path.join(foloder_name, each_file))

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        print("Done loading data. Len of images:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        instance_image = Image.open(img_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        instance_images = self.image_transforms(instance_image)

        caption_path = img_path.replace(".jpg", ".txt")
        with open(caption_path, "r") as f:
            caption = f.read()
            input_ids = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="np",
            ).input_ids
        return Instance(
            pixel_values=DistTensorData(instance_images.to(dtype=flow.float32)),
            input_ids=DistTensorData(flow.tensor(input_ids[0])),
        )
