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
            from projects.MT5.utils.mt5_loader import T5LoaderHuggerFace

            model_loader = T5LoaderHuggerFace(
                libai_cfg_model,
                libai_cfg_model.cfg,
                model_path,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                embedding_dropout_prob=0.0,
            )
            return model_loader.load()
        elif mode == "libai":
            from projects.MT5.utils.mt5_loader import T5LoaderLibai

            model_loader = T5LoaderLibai(
                libai_cfg_model,
                libai_cfg_model.cfg,
                model_path,
            )
            return model_loader.load()
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

    def preprocess(
        self,
        inputs,
        pad: bool = False,
        **kwargs,
    ) -> dict:
        # tokenizer encoder
        encoder_ids = self.tokenizer.encode(inputs, return_tensors="of", is_global=True)

        encoder_input_dict = {
            "encoder_ids": encoder_ids,
        }

        return encoder_input_dict

    def forward(self, encoder_input_dict, **kwargs) -> dict:
        outputs = self.model.generate(encoder_input_dict["encoder_ids"], **kwargs)
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
        "/path/to/libai/projects/MT5/configs/t5_inference.py",
        data_parallel=1,
        tensor_parallel=2,
        pipeline_parallel=2,
        pipeline_stage_id=[0] * 12 + [1] * 12,
        pipeline_num_layers=12 * 2,
        model_path="/path/to/t5-base",
        mode="huggingface",
    )

    text = ["summarize: She is a student, She is tall, She loves study"]
    dict1 = pipeline(text)
    if dist.is_main_process():
        print(dict1)
