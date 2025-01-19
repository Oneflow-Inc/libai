import oneflow.nn as nn
from libai.models.bert_model import BertModel

class BERT_Classifier(nn.Module):
    def __init__(self, cfg):
        label_num = 49
        super().__init__()
        self.encoder = BertModel(cfg)
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.fc = nn.Linear(768, label_num)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        x = encoder_output[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)

        if tokentype_ids is not None and self.training:
            losses = self.criterion(x, tokentype_ids)
            return {"losses": losses}
        else:
            return {"prediction_scores": x}
