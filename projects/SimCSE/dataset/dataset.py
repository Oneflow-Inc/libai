import random
import csv
from oneflow.utils.data import Dataset, DataLoader
from libai.tokenizer import BertTokenizer
import oneflow as flow
from libai.data.structures import DistTensorData, Instance


def load_data(name, path):
    assert name in ['nli', 'wiki', 'sts']
    
    def load_nli_data(path):
        csv_file = csv.reader(open(path, 'r', encoding='utf8'))
        data = [i for i in csv_file][1:]
        random.shuffle(data)
        spl = round(len(data)*0.95)
        return data[:spl], data[spl:]
    
    def load_wiki_data(path):
        data = []
        with open(path, 'r', encoding='utf8') as file:
            for line in file.readlines():
                line = ' '.join(line.strip().split())
                data.append(line)
        random.shuffle(data)
        spl = round(len(data)*0.95)
        return data[:spl], data[spl:]

    def load_sts_data(path):
        data = []
        with open(path, 'r', encoding='utf8') as file:
            for line in file.readlines():
                line = line.strip().split('\t')
                data.append(line)
        return data
    
    if name == 'nli':
        return load_nli_data(path)
    elif name == 'wiki':
        return load_wiki_data(path)
    else:
        return load_sts_data(path)


class TrainDataset(Dataset):
    def __init__(self, name, path):
        self.data = load_data(name, path)
    
    def __len__(self):
        return len(self.data)
    
    def text2id(self, text):
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        ids = [101] + ids + [102]
        attention_mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        sample = Instance(
            input_ids=DistTensorData(ids),
            attention_mask=DistTensorData(attention_mask),
            tokentype_ids=DistTensorData(token_type_ids),
        )
        return sample
    
    def __getitem__(self, index):
        return self.text2id(self.data[index])


class TestDataset(Dataset):
    def __init__(self, name, path):
        self.data = load_data(name, path)
    
    def __len__(self):
        return len(self.data)
    
    def text2id(self, text):
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        length = len(ids)
        ids = [101] + ids + [102] + [0]*(50-length)
        ids = flow.tensor(ids).long()

        attention_mask = [1] * len(ids) + [0]*(50-length)
        attention_mask = flow.tensor(attention_mask).long()

        token_type_ids = [0] * len(ids) + [0]*(50-length)
        token_type_ids = flow.tensor(token_type_ids).long()

        sample = Instance(
            input_ids=DistTensorData(ids),
            attention_mask=DistTensorData(attention_mask),
            tokentype_ids=DistTensorData(token_type_ids),
        )
        return sample

    def __getitem__(self, index):
        # sent1, sent2
        return self.text2id(self.data[index][1]), self.text2id(self.data[index][2]), float(data[index][0])


class PadBatchData:
    def __init__(self):
        self.pad_id = 0
    
    def __call__(self, batch):
        res = dict()
        max_len = max([len(i['input_ids']) for i in batch])
        res['input_ids'] = [i['input_ids'] + [self.pad_id]*(max_len-len(i["input_ids"])) for i in batch]
        res['input_ids'] = [[i, i] for i in res['input_ids']]
        res['input_ids'] = flow.tensor(res['input_ids']).long()
        
        res['attention_mask'] = [i['attention_mask'] + [self.pad_id]*(max_len-len(i["attention_mask"]))  for i in batch]
        res['attention_mask'] = [[i, i] for i in res['attention_mask']]
        res['attention_mask'] = flow.tensor(res['attention_mask']).long()
        
        res['token_type_ids'] = [i['token_type_ids'] + [self.pad_id]*(max_len-len(i["token_type_ids"]))  for i in batch]
        res['token_type_ids'] = [[i, i] for i in res['token_type_ids']]
        res['token_type_ids'] = flow.tensor(res['token_type_ids']).long()
        return res