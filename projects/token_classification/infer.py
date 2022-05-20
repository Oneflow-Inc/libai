from cProfile import label
import sys
import oneflow as flow
from libai.config import LazyConfig
from libai.config import get_config
from libai.tokenizer import BertTokenizer
from libai.config import LazyCall
from libai.engine.default import DefaultTrainer
from libai.utils.checkpoint import Checkpointer
from libai.data.structures import DistTensorData

from model.model import ModelForSequenceClassification
from dataset import CnerDataset
import pdb
from dataset.utils import InputFeatures
from dataset.cner_dataset import cner_processors
import oneflow.nn.functional as F


def get_global_tensor(rawdata):
    t = flow.tensor(rawdata, dtype=flow.long).unsqueeze(0)
    dtd = DistTensorData(t)
    dtd.to_global()
    return dtd.tensor

class GeneratorForEager:
    def __init__(self, config_file, checkpoint_file, vocab_file):
        cfg = LazyConfig.load(config_file)
        cfg.model._target_='model.model.ModelForSequenceClassification'
        processor = cner_processors['cner']()
        label_list = processor.get_labels()
        self.label_map = {label: i for i, label in enumerate(label_list)}
        
        self.model = DefaultTrainer.build_model(cfg).eval()
        Checkpointer(self.model).load(checkpoint_file)
        self.tokenizer = BertTokenizer(vocab_file)
    
    def infer(self, sentence):
        # Encode
        # sentence = " ".join([word for word in sentence])
        tokens_list = self.tokenizer.tokenize(sentence)
        # pdb.set_trace()
        tokens_list= ['[CLS]']+tokens_list + ['[SEP]']
        
        encoder_ids_list =self.tokenizer.convert_tokens_to_ids(tokens_list)
        seq_len = len(encoder_ids_list)

        encoder_input_ids = get_global_tensor(encoder_ids_list)
        
        attention_mask=[0] * seq_len
        token_type_ids= [0] * seq_len
        attention_mask=get_global_tensor(attention_mask)

        token_type_ids=get_global_tensor(token_type_ids)
        encoder_states = self.model(encoder_input_ids,attention_mask,token_type_ids,None)
        encoder_states=encoder_states['prediction_scores']
    
        logits = F.softmax(encoder_states,dim=2)
        logits_label = flow.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]
        self.label_map = {v:k for k,v in self.label_map.items()}
        result=[]
        words=[]
        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]

        for word in sentence:
            words.append(word)

        for label in logits_label:
            result_a=self.label_map[label]
            result.append(result_a)
        output = [{"word":word,"tag":label,"confidence":confidence} for word,label,confidence in zip(words,result,logits_confidence)]
        # pdb.set_trace()
        return output
        
        
        

if __name__ == "__main__":
    config_file = "/workspace/CQL_BERT/libai/output/benchmark/token/config.yaml"
    checkpoint_file = "/workspace/CQL_BERT/libai/output/benchmark/token/model_final"
    vocab_file = "/workspace/CQL_BERT/libai/projects/QQP/QQP_DATA/bert-base-chinese-vocab.txt"
    
    generator = GeneratorForEager(config_file, checkpoint_file, vocab_file)

    sentence = input("输入：\n")
    result = generator.infer(sentence)
 
    print(*result)
