import json
import oneflow as flow
from oneflow.utils.data import Dataset
from libai.data.structures import DistTensorData, Instance
from libai.tokenizer import BertTokenizer


class ExtractionDataSet(Dataset):
    def __init__(self, data_path, vocab_path, id2rel_dict, is_train, indc=None):
        if is_train:
            self.train = True
            data_path = data_path + "/train.json"
        else:
            data_path = data_path + "/dev.json"

        data = load_data(data_path=data_path, vocab_path=vocab_path, id2rel_dict=id2rel_dict, indc=indc)
        self.text = [t.numpy() for t in data['text']]
        self.mask = [t.numpy() for t in data['mask']]
        self.label = [t for t in data['label']]

    def __getitem__(self, idx):
        sample = Instance(
            input_ids=DistTensorData(
                flow.tensor(self.text[idx][0], dtype=flow.long).clone().detach()
            ),
            attention_mask=DistTensorData(
                flow.tensor(self.mask[idx][0], dtype=flow.long).clone().detach()
            ),
            tokentype_ids=DistTensorData(
                flow.tensor(self.label[idx], dtype=flow.long).clone().detach(),
                placement_idx=-1
            )
        )

        return sample

    def __len__(self):
        return len(self.text)


def get_rel_id_maps(id2rel_dict="id2rel.json"):
    with open(id2rel_dict, "r") as read_content:
        id2rel = json.loads(read_content.read())
    rel2id = {}
    for i in id2rel:
        rel2id[id2rel[i]] = int(i)
    return rel2id


def load_data(data_path="train.json", vocab_path='./bert-base-chinese', id2rel_dict="id2rel.json", indc=None):
    rel2id = get_rel_id_maps(id2rel_dict)
    max_length = 128
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    data = {}
    data['label'] = []
    data['mask'] = []
    data['text'] = []

    with open(data_path, 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        if indc is not None:
            temp = temp[:indc]
        for line in temp:
            dic = json.loads(line)
            if dic['rel'] not in rel2id:
                data['label'].append(0)
            else:
                data['label'].append(rel2id[dic['rel']])

            sent = dic['ent1'] + dic['ent2'] + dic['text']
            indexed_tokens = tokenizer.encode(sent)

            avai_len = len(indexed_tokens)
            while len(indexed_tokens) < max_length:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = flow.tensor(indexed_tokens).long().unsqueeze(0)

            # Attention mask
            att_mask = flow.zeros(indexed_tokens.size()).long()
            att_mask[0, :avai_len] = 1
            data['text'].append(indexed_tokens)
            data['mask'].append(att_mask)

    return data
