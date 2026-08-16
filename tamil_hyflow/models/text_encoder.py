import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .frontend import TamilStructuralFrontend

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class SelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = scores.softmax(dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, self.d_model)
        return self.out(y)

class TextBlock(nn.Module):
    def __init__(self, d_model=512, heads=8, d_ff=1408):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, heads)
        self.n2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.ff(self.n2(x))
        return x

class TamilTextEncoder(nn.Module):
    def __init__(self, layers=6, d_model=512, heads=8, d_ff=1408):
        super().__init__()
        self.frontend = TamilStructuralFrontend(d_model=d_model)
        self.layers = nn.ModuleList([TextBlock(d_model, heads, d_ff) for _ in range(layers)])
        self.norm = RMSNorm(d_model)

    def forward(self, features, mask=None):
        h = self.frontend(*features)
        for layer in self.layers:
            h = layer(h, mask)
        return self.norm(h)
