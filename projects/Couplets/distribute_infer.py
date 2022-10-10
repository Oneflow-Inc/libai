import os
import sys

dir_path = os.path.abspath(os.path.dirname(__file__))  # noqa
sys.path.append(dir_path)  # noqa

import oneflow as flow  # noqa
from dataset.mask import make_sequence_mask  # noqa
from tokenizer.tokenizer import CoupletsTokenizer  # noqa

from libai.data.structures import DistTensorData  # noqa
from libai.inference.basic import BasePipeline  # noqa
from libai.utils import distributed as dist  # noqa


def get_global_tensor(rawdata):
    t = flow.tensor(rawdata, dtype=flow.long).unsqueeze(0)
    dtd = DistTensorData(t)
    dtd.to_global()
    return dtd.tensor


class CoupletPipeline(BasePipeline):
    def _parse_parameters(self, **pipeline_parameters):
        preprocess_params = {**pipeline_parameters}
        forward_params = {}
        postprocess_params = {}

        return preprocess_params, forward_params, postprocess_params

    def update_cfg(
        self,
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
        pipeline_stage_id=None,
        pipeline_num_layers=None,
    ):
        super().update_cfg(
            data_parallel,
            tensor_parallel,
            pipeline_parallel,
            pipeline_stage_id,
            pipeline_num_layers,
        )
        self.cfg.vocab_file = "data_test/couplets/vocab.txt"

    def build_tokenizer(self, cfg):
        return CoupletsTokenizer(cfg.vocab_file)

    def generate(self, sentence):
        # Encode
        sentence = " ".join([word for word in sentence])
        tokens_list = self.tokenizer.tokenize(sentence)
        encoder_ids_list = (
            [self.tokenizer.bos_id]
            + self.tokenizer.convert_tokens_to_ids(tokens_list)
            + [self.tokenizer.eos_id]
        )
        seq_len = len(encoder_ids_list)
        encoder_input_ids = get_global_tensor(encoder_ids_list)
        encoder_states = self.model.encode(encoder_input_ids, None)

        # Decode
        decoder_ids_list = [self.tokenizer.bos_id]
        decoder_input_ids = get_global_tensor(decoder_ids_list)
        for i in range(seq_len + 10):
            mask_array = make_sequence_mask(decoder_ids_list)
            decoder_attn_mask = get_global_tensor(mask_array)
            logits = self.model.decode(decoder_input_ids, decoder_attn_mask, encoder_states, None)
            prob = logits[:, -1]
            _, next_word = flow.max(prob, dim=1)
            next_word = next_word.item()
            decoder_ids_list = decoder_ids_list + [next_word]
            decoder_input_ids = get_global_tensor(decoder_ids_list)
            if next_word == self.tokenizer.eos_id:
                break
        result_tokens_list = self.tokenizer.convert_ids_to_tokens(decoder_ids_list)

        return "".join(result_tokens_list).replace("<bos>", "").replace("<eos>", "")

    def preprocess(self, sentence) -> dict:
        input_dict = {"sentence": sentence}
        return input_dict

    def forward(self, input_dict) -> dict:
        model_output = self.generate(input_dict["sentence"])
        model_out_dict = {"下联": model_output}
        return model_out_dict

    def postprocess(self, model_out_dict) -> dict:
        return model_out_dict


if __name__ == "__main__":

    pipeline = CoupletPipeline(
        "projects/Couplets/configs/config.py",
        data_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=4,
        pipeline_stage_id=None,
        pipeline_num_layers=12,
        model_path="output/couplet/model_final/model",
        mode="libai",
    )

    out = pipeline("滚滚长江东逝水")
    if dist.is_main_process():
        print(out)
