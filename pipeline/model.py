"""The models. LSTM is the workhorse, transformer is the "someday" option."""
import torch
import torch.nn as nn

from dataset import FEAT_DIM, NUM_CLASSES


class LSTMEmotion(nn.Module):
    """BiLSTM over keypoint windows. Small enough to train on anything."""

    def __init__(self, feat_dim=FEAT_DIM, hidden=128, layers=2, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.pooled_dim = hidden * 2  # bidirectional, so x2
        self.proj = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(128, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    # pooled() is split out so train.py can bolt the valence/arousal head
    # onto the same encoding without touching the classifier
    def pooled(self, x):  # (B, T, feat) -> (B, 2*hidden)
        h, _ = self.lstm(self.proj(x))
        return h.mean(dim=1)  # mean over time. tried last-hidden, this was better

    def forward(self, x):
        return self.head(self.pooled(x))


class TransformerEmotion(nn.Module):
    """Tiny transformer encoder. Probably overkill until there's way more data,
    but it's here for when the clip library grows."""

    def __init__(self, feat_dim=FEAT_DIM, d_model=128, heads=4, layers=3,
                 num_classes=NUM_CLASSES, dropout=0.2, max_len=256):
        super().__init__()
        self.pooled_dim = d_model
        self.proj = nn.Linear(feat_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)  # learned pos emb
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4,
                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, num_classes)

    def pooled(self, x):
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        return self.encoder(h).mean(dim=1)

    def forward(self, x):
        return self.head(self.pooled(x))


def build_model(name: str, **kw) -> nn.Module:
    return {"lstm": LSTMEmotion, "transformer": TransformerEmotion}[name](**kw)
