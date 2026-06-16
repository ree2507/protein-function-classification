import torch
import torch.nn as nn
from torch.utils.data import Dataset

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_int = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
aa_to_int['<PAD>'] = 0
VOCAB_SIZE = len(aa_to_int)
MAX_LEN = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProteinDataset(Dataset):
    def __init__(self, sequences, labels=None, max_len=MAX_LEN):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_int = [aa_to_int.get(aa, 0) for aa in seq[:self.max_len]]
        if len(seq_int) < self.max_len:
            seq_int += [0] * (self.max_len - len(seq_int))
        if self.labels is not None:
            return torch.tensor(seq_int, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
        return torch.tensor(seq_int, dtype=torch.long)


class ProteinLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super(ProteinLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=0.4 if num_layers > 1 else 0)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        fc_input_dim = hidden_dim * 2 * 2
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes))
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.transpose(1, 2)
        max_pooled = self.global_max_pool(lstm_out).squeeze(-1)
        avg_pooled = self.global_avg_pool(lstm_out).squeeze(-1)
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)
        return self.fc(pooled)


_model_cache = {}

def load_lstm(num_classes=6):
    if 'lstm' not in _model_cache:
        model = ProteinLSTM(VOCAB_SIZE, 128, 128, 2, num_classes).to(device)
        model.load_state_dict(torch.load('models/lstm_model_best.pth', map_location=device, weights_only=True))
        model.eval()
        _model_cache['lstm'] = model
    return _model_cache['lstm']
