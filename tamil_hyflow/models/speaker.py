import torch
import torch.nn as nn

class SpeakerReferenceEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Conv1d(1, 64, 5, stride=2, padding=2)),
            nn.SiLU(),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(64, 128, 5, stride=2, padding=2)),
            nn.SiLU(),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(128, 256, 5, stride=2, padding=2)),
            nn.SiLU(),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(256, 512, 5, stride=2, padding=2)),
            nn.SiLU(),
        )
        self.gru = nn.GRU(512, 256, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        self.key = nn.Linear(256, 256)
        self.value = nn.Linear(256, 256)
        self.out = nn.Linear(256, out_dim)

    def forward(self, audio):
        h = self.convs(audio).transpose(1, 2)
        h, _ = self.gru(h)
        q = self.query.expand(h.shape[0], -1, -1)
        scores = torch.matmul(q, self.key(h).transpose(-2, -1)) / 16.0
        weights = scores.softmax(dim=-1)
        pooled = torch.matmul(weights, self.value(h)).squeeze(1)
        return self.out(pooled)
