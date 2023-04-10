#!/usr/bin/env bash

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export CLASS_DIR="/home/chengpeng/chengpeng/diffusers-pytorch/examples/dreambooth/prior_dog/"
export CLASS_PROMPT="a photo of dog"

python3 projects/Stable_Diffusion/generate_prior_image.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--class_data_dir=$CLASS_DIR \
--class_prompt="$CLASS_PROMPT" \
--num_class_images=200 \
