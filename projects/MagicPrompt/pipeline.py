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
            from libai.models.utils import GPT2LoaderHuggerFace

            model_loader = GPT2LoaderHuggerFace(
                libai_cfg_model,
                libai_cfg_model.cfg,
                model_path,
            )
            model = model_loader.load()
            model.eval()
            return model

        elif mode == "libai":
            from libai.models.utils import GPT2LoaderLiBai

            model_loader = GPT2LoaderLiBai(
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
        # tokenizer encoder
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "[PAD]"
        inputs = self.tokenizer(inputs, return_tensors="of", padding=True)
        inputs = {
            "input_ids": inputs.input_ids,
        }

        return inputs

    def forward(self, inputs, **kwargs) -> dict:
        outputs = self.model.generate(inputs["input_ids"], max_length=50, **kwargs)
        return {"return_ids": outputs}

    def postprocess(self, model_output_dict, **kwargs) -> dict:
        return_ids = model_output_dict["return_ids"]
        records = [
            {"generated_text": self.tokenizer.decode(return_ids[i], skip_special_tokens=True)}
            for i in range(return_ids.size(0))
        ]
        return records


if __name__ == "__main__":
    pipeline = TextGenerationPipeline(
        "projects/MagicPrompt/configs/gpt2_inference.py",
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
        # pipeline_stage_id=[0] * 6 + [1] * 6,
        # pipeline_num_layers=12,
        model_path="/path/to/oneflow-model",
        mode="libai",
    )

    text = ["a dog", "a cute pig", "a cute girl"]
    output = pipeline(inputs=text)
    if dist.is_main_process():
        print(output)
