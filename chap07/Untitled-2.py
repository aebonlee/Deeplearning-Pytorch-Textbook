import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Basic(nn.Module):
    def __init__(
        self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2
    ):
        super(BasicRNN, self).__init__()
        self.n_layers = n_layers  # RNN 계층에 대한 개수
        self.embed = nn.Embedding(n_vocab, embed_dim)  # 워드 임베딩
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.RNN(
            embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True
        )
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)  # 문자를 숫자/벡터로 변환
        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.rnn(x, h_0)
        h_t = x[:, -1, :]
        self.dropout(h_t)  # 드롭아웃
        logit = torch.sigmoid(self.out(h_t))

    def _init_state(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()


nn.Embedding()
