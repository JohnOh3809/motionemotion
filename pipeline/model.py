"""Sequence models for emotion-from-motion classification."""
import torch
import torch.nn as nn

from dataset import FEAT_DIM, NUM_CLASSES


class LSTMEmotion(nn.Module):
    """BiLSTM over keypoint sequences. Small, trains fast, good baseline."""

    def __init__(self, feat_dim=FEAT_DIM, hidden=128, layers=2, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(128, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):                     # x: (B, T, feat_dim)
        h, _ = self.lstm(self.proj(x))        # (B, T, 2*hidden)
        return self.head(h.mean(dim=1))       # temporal mean pool -> (B, C)


class TransformerEmotion(nn.Module):
    """Small Transformer encoder — try when you have more data."""

    def __init__(self, feat_dim=FEAT_DIM, d_model=128, heads=4, layers=3,
                 num_classes=NUM_CLASSES, dropout=0.2, max_len=256):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4,
                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):                     # (B, T, feat_dim)
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        return self.head(self.encoder(h).mean(dim=1))


def build_model(name: str, **kw) -> nn.Module:
    return {"lstm": LSTMEmotion, "transformer": TransformerEmotion}[name](**kw)
