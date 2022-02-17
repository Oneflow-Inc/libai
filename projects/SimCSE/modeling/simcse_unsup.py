import numpy as np
import oneflow as flow
from oneflow import nn
from oneflow.utils.data import Dataset, DataLoader
from libai.tokenizer import BertTokenizer
from config import config
from bert import BertModel
from tqdm import tqdm
from scipy.stats import spearmanr

from dataset import load_data


class CosineSimilarity(nn.Module):
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x, y):
        return flow.sum(x*y, dim=self.dim) / (flow.linalg.norm(x, dim=self.dim) * flow.linalg.norm(y, dim=self.dim))


class MLPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = libai.layers.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            parallel='row',
            layer_idx=-1
        )
        self.activation = libai.layers.build_activation('tanh')
    
    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class SimcseModel(nn.Module):
    def __init__(self, pretrained_model, pooler_type):
        super().__init__()
        self.bert = BertModel(
            vocab_size = config.vocab_size,
            hidden_size = config.hidden_size,
            hidden_layers = config.num_hidden_layers,
            num_attention_heads = config.num_attention_heads,
            intermediate_size = config.intermediate_size,
            hidden_dropout_prob = config.hidden_dropout_prob,
            attention_probs_dropout_prob = config.attention_probs_dropout_prob,
            max_position_embeddings = config.max_position_embeddings
        )
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type
        self.mlp = MLPLayer(config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # train_input_ids:[2*batch_size, seq_len]
        # eval_input_ids:[batch_size, seq_len]
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        last_hidden = outputs[0]    # batch*2, seq_len, hidden
        pooler_output = outputs[1]  # batch*2, hidden
        hidden_states = outputs[2]
        
        if self.pooler_type in ["cls_before_pooler", "cls"]:
            if self.pooler_type == "cls":
                return self.mlp(last_hidden[:, 0])
            # [batch*2, hidden] or [batch, hidden]
            return last_hidden[:, 0]

        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            poolerd_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return poolerd_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            poolerd_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return poolerd_result
        else:
            raise NotImplementedError


def simcse_unsup_loss(y_pred):
    # y_pred: Simcse ouputs, [batch*2, hidden_size]
    y_pred = y_pred.view(-1, 2, y_pred.size(-1))
    z1, z2 = y_pred[:, 0], y_pred[:, 1]
    sim = CosineSimilarity(dim=-1)
    cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
    labels = flow.arange(cos_sim.size(0)).long().to(config.device)
    loss_fc = nn.CrossEntropyLoss()
    loss = loss_fc(cos_sim, labels)
    return loss


def eval(model, dataloder):
    model.eval()
    sim_tensor = flow.tensor([]).to(config.device)
    labels = np.array([])
    with flow.no_grad():
        for source, target, label in dataloder:
            source_input_ids = source.get('input_ids').to(config.device)
            source_attention_mask = source.get('attention_mask').to(config.device)
            source_token_type_ids = source.get('token_type_ids').to(config.device)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids) # [batch, hidden]

            target_input_ids = source.get('input_ids').to(config.device)
            target_attention_mask = source.get('attention_mask').to(config.device)
            target_token_type_ids = source.get('token_type_ids').to(config.device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)

            sim = CosineSimilarity()
            cos_sim = sim(source_pred, target_pred)
            sim_tensor = flow.cat((sim_tensor, cos_sim), dim=0)
            labels = np.append(labels, label)
    return spearmanr(labels, sim_tensor.cpu().numpy()).correlation


def train(model, train_dataloader, dev_dataloader, optimizer):
    model.train()
    global best
    for batch_idx, source in enumerate(tqdm(train_dataloader), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)
        
        out = model(input_ids, attention_mask, token_type_ids)        
        loss = simcse_unsup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0: 
            print(loss.item())    
            # logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval(model, dev_dataloader)
            model.train()
            if best < corrcoef:
                best = corrcoef
                flow.save(model.state_dict(), config.save_path)
                # logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")

if __name__ == "__main__":
    config = config()
    tokenizer = BertTokenizer(config.vocab_path)
    
    train_data = load_data('wiki', path = config.train_data_path)
    dev_data = load_data('sts', path = config.dev_data_path)
    test_data = load_data('sts', path = config.test_data)
    print(len(train_data), len(dev_data), len(test_data))
    
    train_data_loader = DataLoader(Dataset(train_data), batch_size=config.batch_size, collate_fn=PadBatchData)
    dev_data_loader = DataLoader(Dataset(dev_data), batch_size=config.batch_size)
    test_data_loader = DataLoader(Dataset(test_data), batch_size=config.batch_size)
    
    model = SimcseModel()