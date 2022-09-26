import os
import sys

dir_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_path)

import oneflow as flow  # noqa
from dataset.mask import make_padding_mask, make_sequence_mask  # noqa
from modeling.model import Seq2Seq  # noqa
from tokenizer.tokenizer import CoupletsTokenizer  # noqa

from libai.config import LazyConfig  # noqa
from libai.data.structures import DistTensorData  # noqa
from libai.engine.default import DefaultTrainer  # noqa
from libai.utils.checkpoint import Checkpointer  # noqa


def get_global_tensor(rawdata):
    t = flow.tensor(rawdata, dtype=flow.long).unsqueeze(0)
    dtd = DistTensorData(t)
    dtd.to_global()
    return dtd.tensor


class GeneratorForEager:
    def __init__(self, config_file, checkpoint_file, vocab_file):
        cfg = LazyConfig.load(config_file)
        self.model = DefaultTrainer.build_model(cfg).eval()
        Checkpointer(self.model).load(checkpoint_file)
        self.tokenizer = CoupletsTokenizer(vocab_file)

    def infer(self, sentence):
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


if __name__ == "__main__":
    config_file = "output/couplet/config.yaml"
    checkpoint_file = "output/couplet/model_final"
    vocab_file = "data_test/couplets/vocab.txt"
    generator = GeneratorForEager(config_file, checkpoint_file, vocab_file)

    sentence = input("上联：\n")
    result = generator.infer(sentence)
    print("下联：\n" + result)
