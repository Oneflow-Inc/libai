# Stable diffusion 

This is an reimplement of [training stable diffusion](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) in LiBai

## Environment 

Before running the scripts, make sure to install the library's training dependencies:

To make sure you can successfully run the latest versions of the example scripts, we highly recommend installing from source and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. 

### Install libai

libai installation, refer to [Installation instructions](https://libai.readthedocs.io/en/latest/tutorials/get_started/Installation.html)

```bash
# create conda env
conda create -n libai python=3.8 -y
conda activate libai

# install oneflow nightly, [PLATFORM] could be cu117 or cu102
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/[PLATFORM]

# install libai
git clone https://github.com/Oneflow-Inc/libai.git
cd libai
pip install pybind11
pip install -e .
```

- All available `[PLATFORM]`:
  
    <table class="docutils">
    <thead>
    <tr class="header">
    <th>Platform</th>
    <th>CUDA Driver Version</th>
    <th>Supported GPUs</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td>cu117</td>
    <td>&gt;= 450.80.02</td>
    <td>GTX 10xx, RTX 20xx, A100, RTX 30xx</td>
    </tr>
    <tr class="even">
    <td>cu102</td>
    <td>&gt;= 440.33</td>
    <td>GTX 10xx, RTX 20xx</td>
    </tr>
    <tr class="odd">
    <td>cpu</td>
    <td>N/A</td>
    <td>N/A</td>
    </tr>
    </tbody>
    </table></li>

### Install onediff

**Important**


To make sure you can train stable diffusion in LiBai, please install [onediff](https://github.com/Oneflow-Inc/diffusers)

```
git clone https://github.com/Oneflow-Inc/diffusers.git onediff
cd onediff
python3 -m pip install "torch<2" "transformers>=4.26" "diffusers[torch]==0.15.0"
python3 -m pip uninstall accelerate -y
python3 -m pip install -e .
```


Notes

- You need to register a Hugging Face account token and login with `huggingface-cli login`

```bash
python3 -m pip install huggingface_hub
```

- If no command available in the PATH, it might be in the `$HOME/.local/bin`

```bash
 ~/.local/bin/huggingface-cli login
```

## Start training

### Training

- Downloading Demo dataset

    ```shell
    mkdir mscoco && cd mscoco
    wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/libai/Stable_diffusion/00000.tar
    mkdir 00000
    tar -xvf 00000.tar -C 00000/
    ```

- running command

    set your datapath and features in `projects/Stable_Diffusion/configs/config.py`
    ```python
        dataloader.train = LazyCall(build_nlp_train_loader)(
            dataset=[
                # set data path
                LazyCall(TXTDataset)(
                    foloder_name="/path/to/mscoco/00000",
                    ...,
                )
            ]
        )

        train.update(
            dict(
                ...,
                # set checkpointing or not
                activation_checkpoint=dict(enabled=True), # or False

                # set zero stage
                zero_optimization=dict(
                    enabled=True, # or False
                    stage=2, # Highly recommand stage=2, stage=1 or 3 is also supported 
                ),

                # set amp training
                amp=dict(enabled=True), # or False
            )
        )

        # set learning rate 
        optim.lr = 1e-3

    ```

    running with 4 GPU
    ```
    bash tools/train.sh projects/Stable_Diffusion/train_net.py projects/Stable_Diffusion/configs/config.py 4
    ```

### DreamBooth

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.

- Downloading Dataset

    Download images from [here](https://drive.google.com/drive/folders/1BO_dyz-p65qhBRRMRA4TbZ8qW4rB99JZ) and save them in a directory (such as `/path/to/demo_dog/`). This will be our training data.

- DreamBooth Training

    set your datapath and features in `projects/Stable_Diffusion/configs/dreambooth_config.py`
    ```python
        dataloader.train = LazyCall(build_nlp_train_loader)(
            dataset=[
                # set data path
                LazyCall(DreamBoothDataset)(
                    instance_data_root="/path/to/demo_dog/",
                    instance_prompt="a photo of sks dog",
                    ...,
                )
            ]
        )

        train.update(
            dict(
                ...,
                # set checkpointing or not
                activation_checkpoint=dict(enabled=True), # or False

                # set zero stage
                zero_optimization=dict(
                    enabled=True, # or False
                    stage=2, # Highly recommand stage=2, stage=1 or 3 is also supported 
                ),

                # set amp training
                amp=dict(enabled=True), # or False
            )
        )

        # set learning rate 
        optim.lr = 1e-3

    ```

    running with 4 GPU
    ```
    bash tools/train.sh projects/Stable_Diffusion/train_net.py projects/Stable_Diffusion/configs/dreambooth_config.py 4
    ```

- Training DreamBooth with prior-preservation loss

    Prior-preservation is used to avoid overfitting and language-drift. Refer to the paper to learn more about it. For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data. According to the paper, it's recommended to generate `num_epochs * num_samples` images for prior-preservation. 200-300 works well for most cases. 

    - Firstly we need to generate prior-images using the model with a class prompt, here is an example, it will generate 200 prior-images : 
    
        `bash projects/Stable_Diffusion/generate.sh`

        ```shell
        # generate.sh
        export MODEL_NAME="CompVis/stable-diffusion-v1-4" # choose model type
        export CLASS_DIR="/path/to/prior_dog/" # set data save path
        export CLASS_PROMPT="a photo of dog" # set class prompt

        python3 projects/Stable_Diffusion/generate_prior_image.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --class_data_dir=$CLASS_DIR \
        --class_prompt="$CLASS_PROMPT" \
        --num_class_images=200 \   # set num_images
        ```
    
    - Secondly, set your datapath and features in `projects/Stable_Diffusion/configs/prior_preservation_config.py`
        ```python
            dataloader.train = LazyCall(build_nlp_train_loader)(
                dataset=[
                    LazyCall(DreamBoothDataset)(
                        instance_data_root="/path/to/demo_dog/",
                        instance_prompt="a photo of sks dog",
                        class_data_root="/path/to/prior_dog/",
                        class_prompt="a photo of dog",
                        ...,
                    )
                ]
            )

            optim.lr = 2e-6 # set learning rate 
            model.train_text_encoder = True # train text_encoder or not, could be False
            train.train_iter=2000 # set train_iter
            train.log_period=10
        ```

- Training Dreamboth with lora

    Low-Rank Adaption of Large Language Models was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*

    In a nutshell, LoRA allows to adapt pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:
    - Previous pretrained weights are kept frozen so that the model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114)
    - Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.
    - LoRA attention layers allow to control to which extent the model is adapted towards new training images via a `scale` parameter.

    set your datapath and features in `projects/Stable_Diffusion/configs/lora_config.py`
    ```python
        dataloader.train = LazyCall(build_nlp_train_loader)(
            dataset=[
                # set data path
                LazyCall(DreamBoothDataset)(
                    instance_data_root="/path/to/demo_dog/",
                    instance_prompt="a photo of sks dog",
                    ...,
                )
            ]
        )

        train.update(
            dict(
                ...,
                # set checkpointing or not
                activation_checkpoint=dict(enabled=True), # or False

                # set zero stage
                zero_optimization=dict(
                    enabled=True, # or False
                    stage=2, # Highly recommand stage=2, stage=1 or 3 is also supported 
                ),

                # set amp training
                amp=dict(enabled=True), # or False
            )
        )

        # set learning rate 
        optim.lr = 5e-4
    ```

    running with 4 GPU
    ```
    bash tools/train.sh projects/Stable_Diffusion/train_net.py projects/Stable_Diffusion/configs/lora_config.py



## Inference with trained model

model will be saved in `train.output_dir` in `config.py`,

- with lora:
    the model output save dir will be like this:
    ```
    output/stable_diffusion
    ├── config.yaml
    ├── log.txt
    ├── model_sd_for_inference
    │   └── pytorch_lora_weights.bin
    ```
    Here we can use [onediff](https://github.com/Oneflow-Inc/diffusers) to inference our trained lora-model in Libai
    ```python
    import oneflow as flow
    flow.mock_torch.enable()
    from onediff import OneFlowStableDiffusionPipeline
    from typing import get_args
    from diffusers.models.attention_processor import AttentionProcessor

    for processor_type in get_args(AttentionProcessor):
        processor_type.forward = processor_type.__call__

    model_path = "CompVis/stable-diffusion-v1-4"
    pipe = OneFlowStableDiffusionPipeline.from_pretrained(
        model_path,
        use_auth_token=True,
        revision="fp16",
        torch_dtype=flow.float16,
    )
    pipe.unet.load_attn_procs("output/stable_diffusion/model_sd_for_inference/")

    pipe = pipe.to("cuda")

    for i in range(100):
        prompt = "a photo of sks dog"
        with flow.autocast("cuda"):
            images = pipe(prompt).images
            for j, image in enumerate(images):
                image.save(f"{i}.png")
    ```

- without lora
    the model output save dir will be like this:
    ```
    output/stable_diffusion
    ├── config.yaml
    ├── last_checkpoint
    ├── metrics.json
    ├── model_final
    │   ├── graph
    │   ├── lr_scheduler
    │   └── model
    ├── model_sd_for_inference
    │   ├── feature_extractor
    │   │   └── preprocessor_config.json
    │   ├── model_index.json
    │   ├── safety_checker
    │   │   ├── config.json
    │   │   └── pytorch_model.bin
    │   ├── scheduler
    │   │   └── scheduler_config.json
    │   ├── text_encoder
    │   │   ├── config.json
    │   │   └── pytorch_model.bin
    │   ├── tokenizer
    │   │   ├── merges.txt
    │   │   ├── special_tokens_map.json
    │   │   ├── tokenizer_config.json
    │   │   └── vocab.json
    │   ├── unet
    │   │   ├── config.json
    │   │   └── diffusion_pytorch_model.bin
    │   └── vae
    │       ├── config.json
    │       └── diffusion_pytorch_model.bin
    ```

    Here we can use [onediff](https://github.com/Oneflow-Inc/diffusers) to inference our trained model in Libai

    ```python
    import oneflow as flow
    flow.mock_torch.enable()
    from onediff import OneFlowStableDiffusionPipeline

    model_path = "output/stable_diffusion/model_sd_for_inference/"
    pipe = OneFlowStableDiffusionPipeline.from_pretrained(
        model_path,
        use_auth_token=True,
        revision="fp16",
        torch_dtype=flow.float16,
    )

    pipe = pipe.to("cuda")

    for i in range(100):
        prompt = "a photo of sks dog"
        with flow.autocast("cuda"):
            images = pipe(prompt).images
            for j, image in enumerate(images):
                image.save(f"{i}.png")
    ```