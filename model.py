import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BiLSTMAdded(nn.Module):
    """
    모델구조:
        * pretrained BERT -> sequence output를 다음 레이어에 입력으로
        * bidirectional LSTM  -> concat된 forward backward hidden state 를 다음 레이어에 입력으로
        * fully connected
    forward return 형식:
        * dict로 반환 {'logits': output}
    """
    def __init__(self, model_name, num_labels = 30, num_lstm_layers= 2, lstm_dropout= 0.2) -> None:
        """
        model_name: 허깅 페이스 auto model에서 가져올 pretrained 모델의 이름 ex) klue/roberta-large
        num_labels: 분류할 라벨의 수
        num_lstm_layers, lstm_dropout: bidirectional lstm의 레이어수, drop out 비율
        """
        super().__init__()
        self.name = model_name
        self.model_config =  AutoConfig.from_pretrained(self.name)
        self.model_config.num_labels = num_labels
        self.pretrained = AutoModel.from_pretrained(self.name, config=self.model_config)
        self.lstm = nn.LSTM(
            input_size=self.model_config.hidden_size, 
            hidden_size=self.model_config.hidden_size, 
            num_layers= num_lstm_layers, 
            dropout= lstm_dropout,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(self.model_config.hidden_size * 2, num_labels)
    def forward(self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):
        output = self.pretrained(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[0]
        lstm_outputs, (hidden, cell) = self.lstm(output)
        # concat backward and forward hidden state
        output = torch.cat((hidden[0],hidden[1]), dim=1)
        output = self.fc(output)
        return {'logits': output}
