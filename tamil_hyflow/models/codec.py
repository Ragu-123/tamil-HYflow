import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import ResBlock

class ContinuousAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(1, 64, 25, stride=12, padding=12))
        self.conv2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(64, 128, 8, stride=4, padding=2))
        self.conv3 = nn.utils.parametrizations.weight_norm(nn.Conv1d(128, 192, 8, stride=4, padding=2))
        self.conv4 = nn.utils.parametrizations.weight_norm(nn.Conv1d(192, 256, 11, stride=5, padding=3))
        self.resblocks = nn.ModuleList([ResBlock(256) for _ in range(3)])
        self.out = nn.utils.parametrizations.weight_norm(nn.Conv1d(256, 64, 1))

    def forward(self, audio):
        x = F.silu(self.conv1(audio))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = F.silu(self.conv4(x))
        for rb in self.resblocks:
            x = rb(x)
        return self.out(x).transpose(1, 2)
