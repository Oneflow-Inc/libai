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
from libai.config import try_get_key


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
            from projects.Aquila.utils.aquila_loader import AquilaLoaderHuggerFace

            model_loader = AquilaLoaderHuggerFace(
                libai_cfg_model,
                libai_cfg_model.cfg,
                model_path,
            )
            model = model_loader.load()
            model.eval()
            return model

        elif mode == "libai":
            from projects.Aquila.utils.aquila_loader import AquilaLoaderLiBai

            model_loader = AquilaLoaderLiBai(
                libai_cfg_model,
                libai_cfg_model.cfg,
                model_path,
            )
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
        import oneflow as flow
        inputs = flow.tensor(self.tokenizer.encode(inputs, add_bos=True, padding=True))

        inputs = {
            "input_ids": inputs,
        }

        return inputs

    def forward(self, inputs, **kwargs) -> dict:
        inputs = dist.convert_to_distributed_default_setting(inputs["input_ids"])
        outputs = self.model.generate(inputs, max_length=50, **kwargs)
        return {"return_ids": outputs}

    def postprocess(self, model_output_dict, **kwargs) -> dict:
        return_ids = model_output_dict["return_ids"]

        records = [
            {"generated_text": self.tokenizer.decode(return_ids[i])}
            for i in range(return_ids.size(0))
        ]
        return records

    def build_tokenizer(self, cfg):
        tokenizer = None
        if try_get_key(cfg, "tokenization") is not None:
            tokenizer_cfg = cfg.tokenization.tokenizer
            if "vocab_file" not in tokenizer_cfg:
                # If "vocab_file" does not exist in the tokenizer's config,
                # set it to default as f"{model_path}/tokenizer.model"
                tokenizer_cfg.vocab_file = str(
                    Path(self.model_path).joinpath("vocab.json")
                )
            if "merges_file" not in tokenizer_cfg:
                # If "merges_file" does not exist in the tokenizer's config,
                # set it to default as f"{model_path}/tokenizer.model"
                tokenizer_cfg.vocab_file = str(
                    Path(self.model_path).joinpath("merges.txt")
                )
            tokenizer = DefaultTrainer.build_tokenizer(cfg)
        return tokenizer


@click.command()
@click.option(
    "--config_file",
    default="projects/Aquila/configs/aquila_config.py",
    help="Path to the configuration file.",
)
@click.option("--model_path", default=None, help="Path to the model checkpoint.")
@click.option(
    "--mode",
    default="libai",
    help="Mode for the dataloader pipeline, e.g., 'libai' or 'huggingface'.",
)
@click.option(
    "--device", default="cuda", help="Device to run the model on, e.g., 'cuda', 'xpu', 'npu'."
)
def main(config_file, model_path, mode, device):
    pipeline = TextGenerationPipeline(
        config_file,
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
        pipeline_num_layers=32,
        model_path=model_path, #'/root/models/Aquila-7B',
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
