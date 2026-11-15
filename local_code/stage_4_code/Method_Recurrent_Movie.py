"""Movie sentiment networks extending the team's embedding → RNN → linear model."""
import torch
from torch import nn
from local_code.base_class.method import method


class MovieRecurrent(nn.Module):
    def __init__(self, vocab_size, cell='RNN', embedding_dim=128, hidden_size=128,
                 num_layers=1, dropout=0.0, pad_idx=0):
        super().__init__()
        if cell not in ('RNN','LSTM','GRU'):
            raise ValueError('cell must be RNN, LSTM or GRU')
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.recurrent = getattr(nn, cell)(embedding_dim, hidden_size, num_layers=num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        # Unidirectional outputs at true lengths are unaffected by later right padding.
        # Gathering them avoids CPU packed-sequence overhead while keeping padding out
        # of the representation used by the classification head.
        outputs, _ = self.recurrent(self.embedding(x))
        last = outputs[torch.arange(x.size(0), device=x.device), lengths.to(x.device)-1]
        return self.fc(self.dropout(last)).squeeze(1)


class Method_Recurrent_Movie(method):
    """Template MethodModule adapter; the SettingModule owns training/evaluation."""
    def __init__(self, **config):
        super().__init__(config.get('cell', 'RNN'), 'IMDb sentiment classification')
        self.config = config
        self.model = MovieRecurrent(**config)

    def run(self, trainData=None, trainLabel=None, testData=None):
        if testData is None:
            raise ValueError('Pass testData=(tokens, lengths) after training')
        self.model.eval()
        with torch.no_grad():
            return (self.model(*testData) >= 0).long().cpu()
