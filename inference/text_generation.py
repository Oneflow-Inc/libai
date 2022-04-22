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

import sys
from black import T

import numpy as np
import oneflow as flow

sys.path.append(".")  # noqa
from inference.basic import BasePipeline
from libai.data.structures import DistTensorData, Instance
from libai.utils.checkpoint import Checkpointer

class TextGenerationPipeline(BasePipeline):
    def __init__(self, config_file, **kwargs):
        super().__init__(config_file, **kwargs)

    def update_cfg(self):
        self.cfg.model.cfg.bias_dropout_fusion = False
        self.cfg.model.cfg.hidden_layers = 1

    def _parse_parameters(
        self, 
        use_cache=None,
        max_generate_length=10,  
        **pipeline_parameters
    ):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}
        if use_cache is not None:
            assert isinstance(use_cache, bool), "use_cache must be True or False"
            forward_params["use_cache"] = use_cache
        if max_generate_length is not None:
            assert isinstance(max_generate_length, int), "max_generate_length must be integer"
            forward_params["max_generate_length"] = max_generate_length
        return preprocess_params, forward_params, postprocess_params

    def preprocess(
        self,
        inputs,
        pad: bool=False,
        **kwargs,
    ) -> dict:
        # tokenizer encoder
        # inputs = self.tokenizer.tokenize(inputs)
        # input_ids = self.tokenizer.convert_tokens_to_ids(inputs)
        encoder_ids = np.array(self.tokenizer.encode(inputs))
        encoder_padding_mask = self.make_attention_mask(encoder_ids, encoder_ids)   

        encoder_input_dict = {
           "encoder_ids": encoder_ids,
           "encoder_padding_mask": encoder_padding_mask
        }
        
        return encoder_input_dict

    def make_attention_mask(self, source_block, target_block):
        """
        Returns a 2-dimensional (2-D) attention mask
        :param source_block: 1-D array
        :param target_block: 1-D array
        """
        mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
        mask = mask.astype(np.int64)
        return mask

    def make_history_mask(self, block):
        length = block.shape[0]
        arange = np.arange(length)
        history_mask = (
            arange[
                None,
            ]
            < arange[:, None]
        )
        history_mask = history_mask.astype(np.int64)
        return history_mask

    def generate(
        self, 
        encoder_input_dict: dict, 
        use_cache: bool=False,
        max_generate_length: int=10,
        **kwargs
    ) -> dict:
        encoder_nparray_ids = encoder_input_dict["encoder_ids"]
        encoder_nparray_mask = encoder_input_dict["encoder_padding_mask"]
        
        decoder_ids = [self.tokenizer.bos_token_id,]

        for _ in range(max_generate_length):
            # generate decoder input
            decoder_input_ids = decoder_ids[-1:] if use_cache else decoder_ids
            # decoder_input_ids = decoder_ids
            decoder_input_ids = np.array(decoder_input_ids)
            decoder_padding_mask = self.make_attention_mask(decoder_input_ids, decoder_input_ids)
            if not use_cache:
                decoder_padding_mask = decoder_padding_mask * self.make_history_mask(decoder_input_ids)
            encoder_decoder_padding_mask = self.make_attention_mask(
                decoder_input_ids, 
                encoder_nparray_ids
            )

            # set batch size = 1
            encoder_input_ids = flow.tensor(encoder_nparray_ids, dtype=flow.long).unsqueeze(0)
            encoder_padding_mask = flow.tensor(encoder_nparray_mask, dtype=flow.long).unsqueeze(0)
            decoder_input_ids = flow.tensor(decoder_input_ids, dtype=flow.long).unsqueeze(0)
            decoder_padding_mask = flow.tensor(decoder_padding_mask, dtype=flow.long).unsqueeze(0)
            encoder_decoder_padding_mask = flow.tensor(
                encoder_decoder_padding_mask,
                dtype=flow.long
            ).unsqueeze(0)

            # to_global for model input 
            model_input = Instance(
                encoder_input_ids=DistTensorData(encoder_input_ids),
                encoder_attn_mask=DistTensorData(encoder_padding_mask),
                decoder_input_ids=DistTensorData(decoder_input_ids),
                decoder_attn_mask=DistTensorData(decoder_padding_mask),
                encoder_decoder_attn_mask=DistTensorData(encoder_decoder_padding_mask),            
            )

            
            mdoel_input_dict = {
                "use_cache": use_cache
            }
            for key, value in model_input.get_fields().items():
                value.to_global()
                mdoel_input_dict[key] = value.tensor

            # get_next_word
            # change it by yourself according to your needs
            logits = self.model(**mdoel_input_dict)["prediction_scores"]
            prob = logits[:, -1]
            _, next_word = flow.max(prob, dim=1)
            next_word = next_word.item()
            decoder_ids = decoder_ids + [next_word]
            # if next_word == self.tokenizer.eos_token_id:
            #     break
        print(decoder_ids)
        return decoder_ids

    def forward(
        self, 
        encoder_input_dict, 
        use_cache=False,
        max_generate_length=10,
        **kwargs
    ) -> dict:
        self.model.set_cache(encoder_states=None, past_key_values=None)
        decoder_ids = self.generate(encoder_input_dict, use_cache, max_generate_length, **kwargs)
        return {
            "decoder_ids": flow.tensor(decoder_ids)
        }

    def postprocess(
        self,
        model_input_dict,
        **kwargs
    ) -> dict:
        pass
        # prediction_scores = model_input_dict["prediction_scores"]
        # print(prediction_scores)
        # do something according to args


if __name__ == "__main__":
    # for i in range(10):
        print("--------")
        model = TextGenerationPipeline("configs/t5_large_pretrain.py")
        print('---------------------------not cache----------------------------')
        a = model("dog cat you " * 10, use_cache=False, max_generate_length=15)
        print('---------------------------use cache----------------------------')
        a = model("dog cat you " * 10, use_cache=True, max_generate_length=15)
