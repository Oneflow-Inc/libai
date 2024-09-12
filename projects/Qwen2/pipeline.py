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

import oneflow_xpu

from libai.inference.basic import BasePipeline
from libai.utils import distributed as dist

import oneflow
import numpy as np
import threading

global_id = 1
lock = threading.Lock()

def create_save_output_hook(module_name):
    def save_output(module, input, output):
        pass
        # global global_id
        # with lock:
        #     if isinstance(output, tuple):
        #         for idx, out in enumerate(output):
        #             if isinstance(out, oneflow.Tensor):
        #                 np_output = out.numpy()
        #                 np.save(f'/workspace/libai/projects/Qwen2/outputs/{global_id}_{module_name}_{idx}.npy', np_output)
        #     elif isinstance(output, dict):
        #         for idx, out in output.items():
        #             if isinstance(out, oneflow.Tensor):
        #                 np_output = out.numpy()
        #                 np.save(f'/workspace/libai/projects/Qwen2/outputs/{global_id}_{module_name}_{idx}.npy', np_output)
        #     else:
        #         np_output = output.numpy()
        #         np.save(f'/workspace/libai/projects/Qwen2/outputs/{global_id}_{module_name}.npy', np_output)
        #     global_id += 1
    return save_output


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
            from projects.Qwen2.utils.qwen_loader import Qwen2LoaderHuggerFace

            model_loader = Qwen2LoaderHuggerFace(
                libai_cfg_model,
                libai_cfg_model.cfg,
                model_path,
            )
            model = model_loader.load()
            model.eval()
            return model

        elif mode == "libai":
            from projects.Qwen2.utils.qwen_loader import Qwen2LoaderLiBai

            model_loader = Qwen2LoaderLiBai(
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
        import os
        os.makedirs("/workspace/libai/projects/Qwen2/outputs", exist_ok=True)
        for module_name, module in self.model.named_modules():
            if module_name:
                hook = create_save_output_hook(module_name)
                module.register_forward_hook(hook)
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


if __name__ == "__main__":

    pipeline = TextGenerationPipeline(
        "projects/Qwen2/configs/qwen_config.py",
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
        pipeline_num_layers=32,
        device_type='xpu',
        model_path='/root/models/Qwen1.5-7B-Chat',
        mode="huggingface",
    )

    text = [
        "Give three tips for staying healthy.",
    ]
    output = pipeline(inputs=text)
    if dist.is_main_process():
        print(output)
