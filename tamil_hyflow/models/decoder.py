import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=3))
        self.conv2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=3, padding=9))

    def forward(self, x):
        r = x
        x = F.leaky_relu(x, 0.1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv2(x)
        return x + r

class SubbandDecoderBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.utils.parametrizations.weight_norm(nn.Conv1d(64, 256, 7, padding=3))
        self.up1 = nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(256, 256, 11, stride=5, padding=3, output_padding=0))
        self.res1 = nn.ModuleList([ResBlock(256) for _ in range(3)])
        self.up2 = nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(256, 192, 8, stride=4, padding=2, output_padding=0))
        self.res2 = nn.ModuleList([ResBlock(192) for _ in range(3)])
        self.up3 = nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(192, 128, 8, stride=4, padding=2, output_padding=0))
        self.res3 = nn.ModuleList([ResBlock(128) for _ in range(3)])
        self.up4 = nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(128, 64, 24, stride=12, padding=6, output_padding=0))
        self.res4 = nn.ModuleList([ResBlock(64) for _ in range(3)])
        self.post = nn.utils.parametrizations.weight_norm(nn.Conv1d(64, 1, 7, padding=3))

    def forward(self, z):
        x = self.pre(z)
        x = self.up1(x)
        for rb in self.res1:
            x = rb(x)
        x = self.up2(x)
        for rb in self.res2:
            x = rb(x)
        x = self.up3(x)
        for rb in self.res3:
            x = rb(x)
        x = self.up4(x)
        for rb in self.res4:
            x = rb(x)
        return self.post(x)

class MultiBranchSubbandDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = SubbandDecoderBranch()
        self.mid = SubbandDecoderBranch()
        self.high = SubbandDecoderBranch()
        self.fusion = nn.utils.parametrizations.weight_norm(nn.Conv1d(3, 64, 7, padding=3))
        self.res = ResBlock(64)
        self.out = nn.utils.parametrizations.weight_norm(nn.Conv1d(64, 1, 7, padding=3))

    def forward(self, z):
        z = z.transpose(1, 2)
        low = self.low(z)
        mid = self.mid(z)
        high = self.high(z)
        y = torch.cat([low, mid, high], dim=1)
        y = F.silu(self.fusion(y))
        y = self.res(y)
        y = self.out(y)
        return y.squeeze(1), {"low": low.squeeze(1), "mid": mid.squeeze(1), "high": high.squeeze(1)}
