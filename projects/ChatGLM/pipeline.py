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
            from projects.ChatGLM.utils.chatglm_loader import ChatGLMLoaderHuggerFace

            model_loader = ChatGLMLoaderHuggerFace(
                libai_cfg_model,
                libai_cfg_model.cfg,
                model_path,
            )
            model = model_loader.load()
            model.eval()
            return model

        elif mode == "libai":
            from projects.ChatGLM.utils.chatglm_loader import ChatGLMLoaderLiBai

            model_loader = ChatGLMLoaderLiBai(
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

    def preprocess(self, sentence: str, **kwargs) -> dict:
        #
        inputs = {
            "inputs": sentence,
        }
        return inputs

    def forward(self, inputs, **kwargs) -> dict:
        if "history" in kwargs:
            history = kwargs.pop("history")
        else:
            if not hasattr(self, "history"):
                self.history = []
            history = self.history

        response, history = self.model.chat(
            self.tokenizer, inputs["inputs"], history=history, **kwargs
        )
        self.history = history
        return {"response": response, "history": history}

    def postprocess(self, model_output_dict, **kwargs) -> dict:
        return model_output_dict

    def reset_conversation(self):
        self.history = []


if __name__ == "__main__":
    # ----- load huggingface checkpoint -----
    text = "浏览器输入www.baidu.com 并且显示网页，从计算机网络的角度说明实现的全过程"
    glm_model_path = "YOUR_CHATGLM_HUGGINGFACE_PATH"
    pipeline = TextGenerationPipeline(
        "projects/ChatGLM/configs/chatglm_config.py",
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
        pipeline_num_layers=28,
        model_path=glm_model_path,
        mode="huggingface",
    )
    pipeline.model = pipeline.model.half()
    for _ in range(1):
        output = pipeline(inputs=text, do_sample=False, max_length=140)
        if dist.is_main_process():
            print(output["response"])

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(glm_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(glm_model_path, trust_remote_code=True).half().cuda()
    model = model.eval()
    history = []
    for _ in range(1):
        response, history = model.chat(
            tokenizer, text, history=history, do_sample=False, max_length=140
        )
        print(response)

    # ----- load libai checkpoint -----
    # pipeline = TextGenerationPipeline(
    #     "projects/ChatGLM/configs/chatglm_config.py",
    #     data_parallel=1,
    #     tensor_parallel=1,
    #     pipeline_parallel=1,
    #     pipeline_num_layers=28,
    #     model_path="",
    #     mode="libai",
    # )

    # text = ["a dog is flying on the sky", "Wikipedia is a free online", "what is beam search?"]
    # output = pipeline(inputs=text)
    # if dist.is_main_process():
    #     print(output)
