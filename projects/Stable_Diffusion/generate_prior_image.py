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

import argparse
import hashlib
from pathlib import Path

import oneflow as flow
from diffusers import OneFlowStableDiffusionPipeline
from tqdm import tqdm

from projects.Stable_Diffusion.dataset import PromptDataset


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. "
            "Trainable model components should be float32 precision."
        ),
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        required=False,
        help=(
            "Minimal class images for prior preservation loss. "
            "If there are not enough images already present in "
            "class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default="fp16",
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16)."
            " Bf16 requires PyTorch >=1.10.and an Nvidia Ampere GPU. "
            " Default to  fp16 if a GPU is available else fp32."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def main(args):
    class_images_dir = Path(args.class_data_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))
    if args.prior_generation_precision == "fp32":
        torch_dtype = flow.float32
    elif args.prior_generation_precision == "fp16":
        torch_dtype = flow.float16
    elif args.prior_generation_precision == "bf16":
        torch_dtype = flow.bfloat16

    if cur_class_images < args.num_class_images:
        pipeline = OneFlowStableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            use_auth_token=True,
            revision=args.revision,
            torch_dtype=torch_dtype,
        ).to("cuda")
        pipeline.set_progress_bar_config(disable=True)

        num_new_images = args.num_class_images - cur_class_images
        print(f"Number of class images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(args.class_prompt, num_new_images)
        sample_dataloader = flow.utils.data.DataLoader(
            sample_dataset, batch_size=args.sample_batch_size
        )

        for example in tqdm(sample_dataloader, desc="Generating class images"):
            images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = (
                    class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                )
                image.save(image_filename)

        del pipeline
        if flow.cuda.is_available():
            flow.cuda.empty_cache()
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
