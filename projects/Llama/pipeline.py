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

import click
from libai.inference.basic import BasePipeline
from libai.utils import distributed as dist


class TextGenerationPipeline(BasePipeline):
    def load_pretrain_weight(self, libai_cfg_model, model_path, mode="huggingface"):
        """load pretrained model.

        Args:
            libai_cfg_model (libai.models): Lazy config Model in Libai, you can import it
                by `from libai.config.configs.common.models.bert
                    import pretrain_model as libai_cfg_model`
            model_path (str): The directory path of pretrained model,
        """
        if mode == "huggingface":
            from projects.Llama.utils.llama_loader import LlamaLoaderHuggerFace

            model_loader = LlamaLoaderHuggerFace(libai_cfg_model, libai_cfg_model.cfg, model_path,)
            model = model_loader.load()
            model.eval()
            return model

        elif mode == "libai":
            from projects.Llama.utils.llama_loader import LlamaLoaderLiBai

            model_loader = LlamaLoaderLiBai(libai_cfg_model, libai_cfg_model.cfg, model_path,)
            model = model_loader.load()
            model.eval()
            return model

        elif mode == "random":
            from libai.engine import DefaultTrainer

            return DefaultTrainer.build_model(self.cfg)
        else:
            raise NotImplementedError

    def _parse_parameters(self, **pipeline_parameters):
        preprocess_params = {}
        forward_params = {**pipeline_parameters}
        postprocess_params = {}

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs, **kwargs) -> dict:
        # tokenizer encoderW
        inputs = self.tokenizer.tokenize(inputs, add_bos=True, padding=True, device=self.device)
        inputs = {
            "input_ids": inputs,
        }

        return inputs

    def forward(self, inputs, **kwargs) -> dict:
        outputs = self.model.generate(inputs["input_ids"], max_length=50, **kwargs)
        return {"return_ids": outputs}

    def postprocess(self, model_output_dict, **kwargs) -> dict:
        return_ids = model_output_dict["return_ids"]
        records = [
            {"generated_text": self.tokenizer.decode(return_ids[i])}
            for i in range(return_ids.size(0))
        ]
        return records


@click.command()
@click.option(
    "--config_file",
    default="projects/Llama/configs/llama_config.py",
    help="Path to the configuration file.",
)
@click.option("--model_path", default="", help="Path to the model checkpoint.")
@click.option(
    "--mode",
    default="libai",
    help="Mode for the dataloader pipeline, e.g., 'libai' or 'huggingface'.",
)
@click.option(
    "--device", default="cuda", help="Device to run the model on, e.g., 'cuda', 'xpu', 'npu'."
)
def main(config_file, model_path, mode, device):
    if model_path:
        print("Note: The '--model_path' option is for the model checkpoint only. Please configure 'tokenization.tokenizer.pretrained_model_path' directly in the config file.")
    pipeline = TextGenerationPipeline(
        config_file,
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
        pipeline_num_layers=32,
        model_path=model_path,
        mode=mode,
        device=device,
    )

    text = [
        "Give three tips for staying healthy.",
    ]
    output = pipeline(inputs=text)
    if dist.is_main_process():
        print(output)


if __name__ == "__main__":
    main()
