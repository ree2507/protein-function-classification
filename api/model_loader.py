import os
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


class ProteinMultiScaleCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(ProteinMultiScaleCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.spatial_dropout = nn.Dropout1d(p=0.3)

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(128))
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(), nn.BatchNorm1d(128))
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=9, padding=4),
            nn.ReLU(), nn.BatchNorm1d(128))

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.spatial_dropout(x)
        out1, out2, out3 = self.branch1(x), self.branch2(x), self.branch3(x)
        pooled = torch.cat([
            self.global_max_pool(out1).squeeze(-1), self.global_avg_pool(out1).squeeze(-1),
            self.global_max_pool(out2).squeeze(-1), self.global_avg_pool(out2).squeeze(-1),
            self.global_max_pool(out3).squeeze(-1), self.global_avg_pool(out3).squeeze(-1),
        ], dim=1)
        return self.fc(pooled)


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

def load_cnn(num_classes=6):
    if 'cnn' not in _model_cache:
        model = ProteinMultiScaleCNN(VOCAB_SIZE, 64, num_classes).to(device)
        model.load_state_dict(torch.load('models/cnn_model.pth', map_location=device, weights_only=True))
        model.eval()
        _model_cache['cnn'] = model
    return _model_cache['cnn']

def load_lstm(num_classes=6):
    if 'lstm' not in _model_cache:
        model = ProteinLSTM(VOCAB_SIZE, 128, 128, 2, num_classes).to(device)
        model.load_state_dict(torch.load('models/lstm_model_best.pth', map_location=device, weights_only=True))
        model.eval()
        _model_cache['lstm'] = model
    return _model_cache['lstm']

def load_esm2(num_classes=6):
    if 'esm2' not in _model_cache:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel

        ESM2_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)

        class CustomClassifier(nn.Module):
            def __init__(self, hidden_size, num_classes):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, num_classes))
            def forward(self, x):
                x = x[:, 0, :]
                return self.fc(x)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            ESM2_MODEL_NAME, num_labels=num_classes)
        hidden_size = base_model.config.hidden_size
        base_model.classifier = CustomClassifier(hidden_size, num_classes)

        model = PeftModel.from_pretrained(base_model, 'models/esm2_model_best')
        model = model.to(device)
        model.eval()
        _model_cache['esm2'] = (model, tokenizer)
    return _model_cache['esm2']
